# coding=utf-8
"""Classify ROT Breakdown attributes (fields) for social chem.

Based on the `run_glue` huggingface/transformers script (4/7/20 at 3:24pm):
https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

NOTE: This release has not been tested, since the version of huggingface transformers
we used specifies a strict huggingface tokenizers version (0.7.0), which uses rust, and
currently fails to build on a recent rust installation, stopping the whole installation.
However, this file does contain the code we used during development, with minor changes
(e.g., updating dataset filename and format) in hopes it will run. Please report issues
to https://github.com/mbforbes/social-chemistry-101/issues
"""


import argparse
import code
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import random
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter  # type: ignore
from tqdm import tqdm, trange
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    PreTrainedTokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.data_collator import DefaultDataCollator
from transformers.data.processors import DataProcessor, InputExample, InputFeatures


logger = logging.getLogger(__name__)


# These are just for building options for the argument parser. They aren't totally
# correct because we also do multiple choice models.
MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES),
    (),
)

# --------------------------------------------------------------------------------------
# NOTE(Max): New functions / classes for our task below
# --------------------------------------------------------------------------------------

MULTI_LABEL_ATTRS = {"rot-moral-foundations", "rot-categorization"}
MULTIPLE_CHOICE_ATTRS = {"rot-char-targeting", "action-char-involved"}

MULTI_LABEL_LABELS = {
    # multi-label classification
    "rot-moral-foundations": sorted(
        [
            "care-harm",
            "fairness-cheating",
            "loyalty-betrayal",
            "authority-subversion",
            "sanctity-degradation",
        ]
    ),
    "rot-categorization": sorted(
        ["morality-ethics", "social-norms", "advice", "description"]
    ),
}


@dataclass(frozen=True)
class MultipleChoiceInputExample:
    """
    A single training/test example for multiple choice
    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


ROTInputExample = Union[InputExample, MultipleChoiceInputExample]


@dataclass(frozen=True)
class MultipleChoiceInputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


ROTInputFeatures = Union[InputFeatures, MultipleChoiceInputFeatures]


class MultipleChoiceDataset(Dataset):

    features: List[MultipleChoiceInputFeatures]

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i: int) -> MultipleChoiceInputFeatures:
        return self.features[i]


def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean()


def acc_and_f1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    acc = simple_accuracy(preds, labels)
    p, r, f1_micro, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average="micro"
    )
    return {
        "acc": acc,
        "precision": p,
        "recall": r,
        "f1_micro": f1_micro,
        "f1_macro": f1_score(y_true=labels, y_pred=preds, average="macro"),
        "f1_weighted": f1_score(y_true=labels, y_pred=preds, average="weighted"),
        "acc_and_f1_micro": (acc + f1_micro) / 2,
    }


def per_label_metrics(
    preds: np.ndarray, labels: np.ndarray, attr: str
) -> Dict[str, float]:
    # because we only get one-level nesting in tensorboard, doing all acc/f1 metrics for
    # all labels, which would be different for each task, would overwhelm the dashboard.
    # we'll just do accuracy and macro f1 per label in the dashboard for now, as well as
    # all overall stats.
    assert labels.shape[1] == len(MULTI_LABEL_LABELS[attr])
    res = {}
    for i, col in enumerate(MULTI_LABEL_LABELS[attr]):
        res[f"{col}/acc"] = simple_accuracy(preds[:, i], labels[:, i])
        res[f"{col}/f1_macro"] = simple_accuracy(preds[:, i], labels[:, i])
    # also add overall scores
    res["acc"] = simple_accuracy(preds.flatten(), labels.flatten())
    res["f1_macro"] = f1_score(y_true=labels, y_pred=preds, average="macro")
    p, r, f1_micro, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average="micro"
    )
    res["precision"] = p
    res["recall"] = r
    res["f1_micro"] = f1_micro
    return res


def score(logits: np.ndarray, labels: np.ndarray, attr: str) -> Dict[str, float]:
    """
    Scores model output vs labels according to scheme based on attr's task. Returns
    {metric name: value}.

    logits: (B x C) NOT log probs or sigmoids, just model scores in [-inf, inf]
    labels:
        - Multi-class: (B
        - Multi-label: (B x C)
    """
    assert len(logits) == len(labels)
    if attr not in MULTI_LABEL_ATTRS:
        # multi-class classification. softmax and log don't change relative value
        # ordering (for the argmax), so we don't need to do them.
        return acc_and_f1(np.argmax(logits, axis=1), labels)
    else:
        # multi-label classification. we'd do a sigmoid and check if it was closer to 0
        # or 1, but this is equivalent to checking if the raw score is > or < 0.
        return per_label_metrics((logits > 0).astype(int), labels, attr)


class ROTBreakdownProcessor(DataProcessor):
    """Processor for ROT Breakdowns (predicting specific attribute)."""

    # things we cache
    _label_lists: Dict[str, List[str]] = copy.deepcopy(MULTI_LABEL_LABELS)

    def __init__(self, data_dir: str, filename: str) -> None:
        # add dummy labels for now for multiple choice. unclear if these matter.
        for mc_attr in MULTIPLE_CHOICE_ATTRS:
            self._label_lists[mc_attr] = ["0", "1", "2", "3"]
        self._data_path = os.path.join(data_dir, filename)
        self._prefix = "rot-ids"

    def get_train_examples(
        self, attr: str, sit: bool, rot: bool, action: bool, judgment: bool
    ) -> List[ROTInputExample]:
        return self._get_examples("train", attr, sit, rot, action, judgment)

    def get_dev_examples(
        self, attr: str, sit: bool, rot: bool, action: bool, judgment: bool
    ) -> List[ROTInputExample]:
        return self._get_examples("dev", attr, sit, rot, action, judgment)

    def get_test_examples(
        self, attr: str, sit: bool, rot: bool, action: bool, judgment: bool
    ) -> List[ROTInputExample]:
        return self._get_examples("test", attr, sit, rot, action, judgment)

    def get_labels(self, attr: str) -> List[str]:
        # return from cache if it's there (true if any split already loaded)
        if attr in self._label_lists:
            return self._label_lists[attr]
        # else, load and throw away training data, which will cache it
        self._get_examples(
            "train",
            attr,
            True,
            True,
            False,
            False,
            "data/dataset/social-chem-101.v1.0.tsv",
        )
        return self._label_lists[attr]

    @staticmethod
    def _get_label_options(attr: str, df: pd.DataFrame) -> List[str]:
        """Gets list of unique label options for this attr."""
        # for most basic tasks, this is simply the set of unique values. for
        # "choose-0-to-N" tasks, this is more complex.
        if attr not in MULTI_LABEL_ATTRS:
            return sorted(list(pd.unique(df[attr])))
        raise ValueError(f"Unhandled attr '{attr}' for label options")

    def _get_input(self, row: pd.Series, input_key: str) -> str:
        if input_key != "judgment":
            return row[input_key]
        # for judgment, we construct a string
        return {
            -2: "very bad",
            -1: "bad",
            0: "expected / OK",
            1: "good",
            2: "very good",
        }[row["action-moral-judgment"]]

    def _get_chars_label(self, chars_str: str, char_idx: str) -> Tuple[List[str], str]:
        """For multiple choice only. Always gives 4 options.

        chars_str looks lke, e.g.,
            "narrator|him"

        char_idx looks like one of:
            "char-none" <-- "no one listed"
            "char-0"    <-- let's say this was the answer ("narrator")
            "char-1"    <-- "him"
            ...
            "char-5"    <-- n/a in this example

        Returns (
            ['nobody', 'narrator', 'him', ''],
            '1',
        )
        """
        # we collapse characters with index > 2 into a (2+) bin which will just have the
        # name of the 2 char (quite rare, i would guess because of duplicate pronouns)
        chars = ["nobody", "", "", ""]
        for i, char in enumerate(chars_str.split("|")):
            if i + 1 == len(chars):
                break
            chars[i + 1] = char
        assert len(chars) == 4

        # 0 = none
        # 1 = char-0
        # 2 = char-1
        # 3 = char-2, char-3, char-4, char-5
        label = "0"
        if char_idx == "char-none":
            pass
        else:
            idx = int(char_idx.split("-")[1]) + 1
            idx = min(idx, 3)
            label = str(idx)

        return chars, label

    def _get_examples(
        self,
        split: str,
        attr: str,
        sit: bool,
        rot: bool,
        action: bool,
        judgment: bool,
        path_override: Optional[str] = None,
    ) -> List[ROTInputExample]:
        # load full data, filter bad ROTs and get split
        data_path = self._data_path
        if path_override is not None:
            data_path = path_override
        df = pd.read_csv(data_path, sep="\t").convert_dtypes()
        df = df[df["rot-bad"] == 0]
        df = df[df["split"] == split]

        # additional filtering to remove null entries for this column. for
        # "choose-0-to-N" classification, we don't want to remove "n/a", because that
        # actually means that 0 is the correct choice.
        if attr not in MULTI_LABEL_ATTRS:
            logger.info(f"ROT Breakdown: Full split: {len(df)}")
            df = df[~df[attr].isna()]
            # for action-agree, we also want the action-moral-judgment column to exist
            if attr == "action-agree":
                df = df[~df["action-moral-judgment"].isna()]
            logger.info(f"ROT Breakdown: Without N/A for column {attr}: {len(df)}")

        # save set of labels.
        if attr not in self._label_lists:
            self._label_lists[attr] = self._get_label_options(attr, df)

        # figure out input text
        inp_keys = input_identifier(sit, rot, action, judgment).split("-")
        assert len(inp_keys) in [1, 2]

        # construct examples
        res: List[ROTInputExample] = []
        for _, row in df.iterrows():
            if attr in MULTIPLE_CHOICE_ATTRS:
                # multiple choice classification. we always have two input keys.
                chars, label = self._get_chars_label(row["characters"], row[attr])
                ie = MultipleChoiceInputExample(
                    example_id=row["rot-id"],
                    question=row[inp_keys[0]],
                    contexts=[row[inp_keys[1]] for _ in range(len(chars))],
                    endings=chars,
                    label=label,
                )
            else:
                # multi-class and multi-label classification
                ie = InputExample(
                    guid=row["rot-id"],
                    text_a=self._get_input(row, inp_keys[0]),
                    text_b=(
                        self._get_input(row, inp_keys[1])
                        if len(inp_keys) == 2
                        else None
                    ),
                    label=(str(row[attr]) if pd.isna(row[attr]) else row[attr]),
                )
            res.append(ie)
        return res


def input_identifier(
    incl_sit: bool, incl_rot: bool, incl_action: bool, incl_judgment: bool
) -> str:
    """Identify what text is input to the model.

    For caching training data, picking columns of tsv dataset file to load."""
    pieces = []
    if incl_sit:
        pieces.append("situation")
    if incl_rot:
        pieces.append("rot")
    if incl_action:
        pieces.append("action")
    if incl_judgment:
        pieces.append("judgment")
    return "-".join(pieces)


def convert_examples_to_features_multiple_choice(
    examples: List[MultipleChoiceInputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id,
    pad_on_left,
    pad_token,
) -> List[MultipleChoiceInputFeatures]:
    """Sorry for the duplication w/ method below.

    Easier to do this for now than refactor.
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="examples -> features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(
            zip(example.contexts, example.endings)
        ):
            text_a = context
            text_b = example.question + " " + ending
            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs]
            if "attention_mask" in choices_inputs[0]
            else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs]
            if "token_type_ids" in choices_inputs[0]
            else None
        )

        features.append(
            MultipleChoiceInputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


def convert_examples_to_features(
    examples: List[InputExample],
    processor: ROTBreakdownProcessor,
    attr: str,
    tokenizer,
    max_length=512,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        processor: ROTBreakdownProcessor
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        a list of ``InputFeatures`` which can be fed to the model

    """
    logger.info("Using label list %s for attr %s" % (label_list, attr))
    label_map = {label: i for i, label in enumerate(label_list)}

    # record input ids length to figure out on next run what a reasonable max sequence
    # length ought to be.
    input_lengths: typing.Counter[int] = Counter()

    features = []
    skipped = 0
    for (ex_index, example) in enumerate(examples):
        try:
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len_examples))

            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            input_lengths[len(input_ids)] += 1

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length
                )

            # sanity checks
            assert (
                len(input_ids) == max_length
            ), "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert (
                len(attention_mask) == max_length
            ), "Error with input length {} vs {}".format(
                len(attention_mask), max_length
            )
            assert (
                len(token_type_ids) == max_length
            ), "Error with input length {} vs {}".format(
                len(token_type_ids), max_length
            )

            # Label getting is changed for multi-class (default) vs multi-label (new)
            # classification. For multi-class (default), there's a single correct label. For
            # multi-label, we have N label choices, all of which can be on or off (0 or 1).
            label: Union[int, List[int]]
            if attr not in MULTI_LABEL_ATTRS:
                label = label_map[example.label]
            else:
                label = [0] * len(label_map)
                for label_name in example.label.split("|"):
                    if label_name == str(pd.NA):
                        assert label_name == example.label
                        continue
                    elif label_name in label_map:
                        label[label_map[label_name]] = 1
                    else:
                        raise ValueError(
                            f"Unknown label name {label_name} in example label {example.label}"
                        )

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info(
                    f"text_a: {example.text_a} ({len(example.text_a.split(' '))} naive words)"
                )
                logger.info(
                    f"text_b: {example.text_b} "
                    + (
                        f"({len(example.text_b.split(' '))} naive words"
                        if example.text_b is not None
                        else ""
                    )
                )
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
                )
                logger.info(
                    "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
                )
                logger.info(f"label: {example.label} (value = {label})")

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label,
                )
            )
        except Exception as exp:
            skipped += 1

    print(f"Skipped {skipped}/{len(examples)} due to errors (likely missing attrs)")
    # display_input_lengths(input_lengths)

    return features


def get_hparam_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """A little bit of massaging to avoid errors."""
    res = {}
    allowed_types = [str, bool, int, float, torch.Tensor]
    for k, v in vars(args).items():
        if sum([isinstance(v, t) for t in allowed_types]) == 0:
            v = str(v)
        res[k] = v
    return res


# --------------------------------------------------------------------------------------
# NOTE(Max): New functions / classes for our task above
# --------------------------------------------------------------------------------------


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_loss(logits, labels, attr: str) -> Any:
    """Since we also do multi-label classification, loss isn't always CrossEntropy.

    NOTE: That "logits" are raw values output from the model. These are not log
        probabilities. log and softmax or sigmoid happens in the loss function.

    Logits: B x C
    Labels:
        - Multi-class: B
        - Multi-label: B x C
    """
    if attr not in MULTI_LABEL_ATTRS:
        # No viewing necessary, I think.
        return nn.CrossEntropyLoss()(logits, labels)
    else:
        return nn.BCEWithLogitsLoss()(logits, labels)


def train(args, train_dataset, model, tokenizer):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        # Name the writer (directory) something more useful. This may not be robust if
        # args.model_name_or_path is, in fact, a path. But it's way more helpful for
        # right now.
        t = datetime.strftime(datetime.now(), "%m%d_%H%m%S")
        tb_writer = SummaryWriter(
            log_dir=f"runs/{t}_{args.attr}_{args.model_name_or_path}"
        )
        tb_writer.add_text("task/attr", args.attr)
        tb_writer.add_text("task/model", args.model_name_or_path)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    collate_fn = (
        DefaultDataCollator().collate_batch
        if args.attr in MULTIPLE_CHOICE_ATTRS
        else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs   = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            # Prepare batch. This has diverged for multiple choices and other configs.
            if args.attr in MULTIPLE_CHOICE_ATTRS:
                # multiple choice.
                batch = {k: v.to(args.device) for k, v in batch.items()}
                # NOTE: Not handing segment ids right now.
                if args.model_type in {"bert", "xlnet", "albert"}:
                    raise ValueError(
                        f"Need to implement segment ids for model type {args.model_type}"
                    )
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": None,
                }
                labels = batch["labels"]
            else:
                # multi-class, multi-label classification
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                labels = batch[3]
                # Many model types don't use segment_ids. And it looks like maybe we have to
                # "not include" them in different ways.
                # (XLM, DistilBERT, RoBERTa, XLM-RoBERTa, BART)
                if args.model_type not in ["distilbert", "bart"]:
                    inputs["token_type_ids"] = (
                        batch[2]
                        if args.model_type in ["bert", "xlnet", "albert"]
                        else None
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            logits = outputs[0]
            loss = get_loss(logits, labels, args.attr)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Maybe compute and report training metrics per batch.
                if args.train_metrics and global_step % 100 == 0:
                    train_metrics = score(
                        logits.detach().cpu().numpy(),
                        labels.detach().cpu().numpy(),
                        args.attr,
                    )
                    for key, val in train_metrics.items():
                        tb_writer.add_scalar(f"train/{key}", val, global_step)

                # Maybe eval and log eval metrics + lr + loss.
                # NOTE(max): added global_step == 1 for eval @ start.
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and (
                        global_step % args.logging_steps == 0
                        or (args.evaluate_at_start and global_step == 1)
                    )
                ):
                    logs = {}
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            logs[f"eval/{key}"] = value

                    # Training metrics don't make sense for the "eval at start" hack.
                    if global_step > 1:
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["training/learning_rate"] = learning_rate_scalar
                        logs["train/loss"] = loss_scalar
                        logging_loss = tr_loss

                    # Write out whatever we've gotten.
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.warning(
                        f"{args.attr} ({args.model_name_or_path}) [@ {global_step}]"
                    )
                    logger.warning(json.dumps({**logs, **{"step": global_step}}))

                # Maybe save model checkpoint
                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):

                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # Run final eval at end of training for results to save w/ hyperparams.
    # NOTE: could args.local_rank == 0 be OK too?
    if args.local_rank == -1 and args.evaluate_during_training:
        results = evaluate(args, model, tokenizer)
        tb_writer.add_hparams(
            get_hparam_dict(args), {f"hparam/eval/{k}": v for k, v in results.items()}
        )

    # always close tensorboard writer
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", test=False):
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, test=test)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = (
        DefaultDataCollator().collate_batch
        if args.attr in MULTIPLE_CHOICE_ATTRS
        else None
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
    )

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    all_logits = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            if args.attr in MULTIPLE_CHOICE_ATTRS:
                # multiple choice.
                batch = {k: v.to(args.device) for k, v in batch.items()}
                # NOTE: Not handing segment ids right now.
                if args.model_type in {"bert", "xlnet", "albert"}:
                    raise ValueError(
                        f"Need to implement segment ids for model type {args.model_type}"
                    )
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": None,
                }
                labels = batch["labels"]
            else:
                # multi-class, multi-label classification
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                labels = batch[3]
                # Many model types don't use segment_ids. And it looks like maybe we have to
                # "not include" them in different ways.
                # (XLM, DistilBERT, RoBERTa, XLM-RoBERTa, BART)
                if args.model_type not in ["distilbert", "bart"]:
                    inputs["token_type_ids"] = (
                        batch[2]
                        if args.model_type in ["bert", "xlnet", "albert"]
                        else None
                    )

            outputs = model(**inputs)
            logits = outputs[0]
            tmp_eval_loss = get_loss(logits, labels, args.attr)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # accumulate precitions and labels
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0
            )
    eval_loss = eval_loss / nb_eval_steps

    # compute metrics
    results = score(all_logits, out_label_ids, args.attr)

    # add what the task was to results
    results["attr"] = args.attr

    # report
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    # write
    output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.jsonl")
    logger.info(f"Writing results to {output_eval_file}")
    with open(output_eval_file, "a+") as f:
        f.write(json.dumps(results) + "\n")

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, test=False) -> Dataset:
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()  # type: ignore

    split = "train"
    if evaluate:
        split = "test" if test else "dev"

    processor = ROTBreakdownProcessor(args.data_dir, args.data_filename)
    # Load data features from cache or dataset file
    # TODO: model identifier in cache path
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            split,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.attr),
            input_identifier(
                args.input_situation,
                args.input_rot,
                args.input_action,
                args.input_judgment,
            ),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        # NOTE(max): This assumes 2 labels :-(
        # if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        #     # HACK(label indices are swapped in RoBERTa pretrained model)
        #     label_list[1], label_list[2] = label_list[2], label_list[1]

        get_example_fn = {
            "train": processor.get_train_examples,
            "dev": processor.get_dev_examples,
            "test": processor.get_test_examples,
        }[split]
        examples = get_example_fn(
            args.attr,
            args.input_situation,
            args.input_rot,
            args.input_action,
            args.input_judgment,
        )
        label_list = processor.get_labels(args.attr)
        if args.attr in MULTIPLE_CHOICE_ATTRS:
            features = convert_examples_to_features_multiple_choice(
                examples=examples,
                label_list=label_list,
                max_length=args.max_seq_length,
                tokenizer=tokenizer,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.pad_token_id,
            )
        else:
            features = convert_examples_to_features(
                examples,
                processor,
                args.attr,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # xlnet: pad on left
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
            )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()  # type: ignore

    # Build dataset. For multi-class and multi-label classification, we convert to
    # tensors. For multiple choice, I guess this happens later!
    dataset: Dataset
    if args.attr in MULTIPLE_CHOICE_ATTRS:
        # multiple choice
        dataset = MultipleChoiceDataset(features)
    else:
        # multi-class and multi-label classification
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )

        # BCE requires float labels.
        label_dtype = torch.long if args.attr not in MULTI_LABEL_ATTRS else torch.float
        all_labels = torch.tensor([f.label for f in features], dtype=label_dtype)

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
    return dataset


def main():
    # logging hacks, i'm sorry
    logname2level = {name: level for level, name in logging._levelToName.items()}

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--attr",
        default=None,
        type=str,
        required=True,
        help="ROT Breakdown attribute to predict (columns of ROT Breakdown file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written. If not pro",
    )
    parser.add_argument(
        "--input_situation",
        action="store_true",
        help="Whether to include the situation as input to the model (inputs: min 1, max 2)",
    )
    parser.add_argument(
        "--input_rot",
        action="store_true",
        help="Whether to include the ROT as input to the model (inputs: min 1, max 2)",
    )
    parser.add_argument(
        "--input_action",
        action="store_true",
        help="Whether to include the action as input to the model (inputs: min 1, max 2)",
    )
    parser.add_argument(
        "--input_judgment",
        action="store_true",
        help="Whether to include the action's (moral) judgment (discrete) as input to the model (inputs: min 1, max 2)",
    )

    # Defaults I've added
    parser.add_argument(
        "--data_dir",
        default="data/dataset/",
        type=str,
        help="Directory containing the Social Chem 101 dataset .tsv file.",
    )
    parser.add_argument(
        "--data_filename",
        default="social-chem-101.v1.0.tsv",
        type=str,
        help="Filename to load in data_dir. Should be a social chem .tsv file.",
    )

    # Other parameters
    parser.add_argument(
        "--train_metrics",
        action="store_true",
        help="Whether to report metrics on train set during training.",
    )
    parser.add_argument(
        "--logging_level",
        default="INFO",
        choices=logname2level.keys(),
        type=str,
        help="Logging level to use on main process.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set entirely separately from training.",
    )
    parser.add_argument(
        "--evaluate_at_start",
        action="store_true",
        help="Whether to run eval at start of training.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step, as well as after training.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=4.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=250,
        help="Evaluate on entire dev set and write all dashboard metrics every X update steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    args = parser.parse_args()

    # check input settings. need minimum 1, maximum 2 text inputs.
    n_inputs = sum(
        [args.input_situation, args.input_rot, args.input_action, args.input_judgment]
    )
    if n_inputs not in [1, 2]:
        raise ValueError(
            f"Need 1 -- 2 of --input_situation, --input_rot, --input_action, --input_judgment, but got {n_inputs}."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logname2level[args.logging_level]
        if args.local_rank in [-1, 0]
        else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare ROT Breakdown task :-)
    processor = ROTBreakdownProcessor(args.data_dir, args.data_filename)
    label_list = processor.get_labels(args.attr)
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        # NOTE: may want to remove or `num_labels` for multiple choice?
        num_labels=num_labels,
        # finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_builder = (
        AutoModelForMultipleChoice
        if args.attr in MULTIPLE_CHOICE_ATTRS
        else AutoModelForSequenceClassification
    )
    model = model_builder.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        results = evaluate(args, model, tokenizer, test=True)

    return results


if __name__ == "__main__":
    main()
