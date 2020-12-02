"""Data loading library functions, common across models (and elsewhere)."""

from collections import OrderedDict, Counter
import itertools
import logging
import os
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union

import IPython
import numpy as np
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
from tqdm import tqdm


AGREE_TO_STR = OrderedDict(
    [(0, "nobody"), (1, "rare"), (2, "controversial"), (3, "most"), (4, "all")]
)
ACTION_MORAL_JUDGMENT_TO_STR = OrderedDict(
    [(-2, "very-bad"), (-1, "bad"), (0, "ok"), (1, "good"), (2, "very-good")]
)
ACTION_PRESSURE_TO_STR = OrderedDict(
    [
        (-2, "strong-against"),
        (-1, "against"),
        (0, "discretionary"),
        (1, "for"),
        (2, "strong-for"),
    ]
)

# reverse maps for going backwards

STR_TO_AGREE = {v: k for k, v in AGREE_TO_STR.items()}
STR_TO_ACTION_MORAL_JUDGMENT = {v: k for k, v in ACTION_MORAL_JUDGMENT_TO_STR.items()}
STR_TO_ACTION_PRESSURE = {v: k for k, v in ACTION_PRESSURE_TO_STR.items()}

DEMO_ATTR_ORDER = [
    # RoT
    "rot-char-targeting",
    "rot-categorization",
    "rot-moral-foundations",
    "rot-agree",
    # action
    "action-agency",
    "action-moral-judgment",
    "action-agree",
    "action-legal",
    "action-pressure",
    "action-hypothetical",
]


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def flatten(l: List[Union[Any, List[Any]]]) -> List[Any]:
    """Flattens 1d and 2d and mixed 1d/2d lists into 1d list."""
    return list(itertools.chain.from_iterable(l))


def init_model(model_name: str, args):
    """
    Initialize a pre-trained LM
    :param model_name: the LM name or path
    :param args: including device and do_lower_case
    :return: the model and tokenizer
    """
    try:
        do_lower_case = args.do_lower_case
    except:
        do_lower_case = False

    # Pretrained-model
    if "random_" not in model_name or (
        "random_" in model_name and os.path.exists(model_name)
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=do_lower_case
        )
        model = AutoModelWithLMHead.from_pretrained(model_name)
    else:
        base_model = model_name.replace("random_", "")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, do_lower_case=do_lower_case
        )
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)
    model.eval()
    return tokenizer, model


def get_rot_attributes(row: pd.Series) -> List[str]:
    """
    Gets a row from the rot-details tsv file and returns a list of string rot-related attributes
    :param row: dataframe row
    :return: a list of string rot-related attributes
    """
    return (
        get_attr(row, "rot-categorization")
        + get_attr(row, "rot-moral-foundations")
        + get_attr(row, "rot-agree")
    )


def get_action_attributes(row: pd.Series) -> List[str]:
    """
    Gets a row from the rot-details tsv file and returns a list of string action-related attributes
    :param row: dataframe row
    :return: a list of string action-related attributes
    """
    return (
        get_attr(row, "action-agency")
        + get_attr(row, "action-moral-judgment")
        + get_attr(row, "action-agree")
        + get_attr(row, "action-legal")
        + get_attr(row, "action-pressure")
        + get_attr(row, "action-hypothetical")
    )


def get_all_attributes(df: pd.DataFrame) -> List[str]:
    """
    Extract the tokens of the various attributes to add special tokens.

    Note that "duplicates" are removed (e.g., if two tokens have the same value, like
    with rot agreement and action agreement).
    """
    tokens = []

    # Use values as is
    for col in ["action-agency", "action-legal", "action-hypothetical"]:
        tokens.extend([f"<{val}>" for val in df[col].unique() if pd.notna(val)])

    # Map values
    for col, curr_map in [
        ("action-moral-judgment", ACTION_MORAL_JUDGMENT_TO_STR),
        ("action-agree", AGREE_TO_STR),
        ("action-pressure", ACTION_PRESSURE_TO_STR),
        ("rot-agree", AGREE_TO_STR),
    ]:
        tokens.extend(
            [f"<{curr_map[val]}>" for val in df[col].unique() if pd.notna(val)]
        )

    # Lists of values
    for col in ["rot-categorization", "rot-moral-foundations"]:
        values = [val for val in df[col].unique() if pd.notna(val)]
        for val_list in values:
            tokens.extend([f"<{val}>" for val in val_list.split("|")])

    return list(set(tokens))


def load_data(
    in_file: str,
    value_to_predict: str,
    input_type: str = "situation_and_attributes",
    output_type: str = "action_or_rot",
    split: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Loads the dataset file in the format for training the generative model, i.e.:

    If predicting the RoT, e.g.:

    Input: <situation> [attrs] <rot-categorization> <rot-moral-foundations> <rot-agree>
    Output: [rot] <rot>

    Else, if predicting the action, e.g.:

    <situation> [attrs] <action-agency> <action-moral-judgment> <action-agree>
    <action-legal> <action-pressure> <action-hypothetical> [action]
    Output: <action>

    Args:
        - in_file: TSV rot-details file
        - value_to_predict: rot / action / demo
        - input_type: situation_and_attributes | situation | all (multi-task); ignored
            if value_to_predict == "demo"
        - output_type: action_or_rot | action_or_rot_and_attributes | attributes | all
            (multi-task); ignored if value_to_predict == "demo"

    Returns a list of tuples (input, output)
    """
    df = pd.read_csv(in_file, delimiter="\t").convert_dtypes()

    # Take only the current split
    if split is not None:
        df = df[df["split"] == split]

    num_before = len(df)

    # Remove bad ROTs, and any other filtering / processing.
    df = df[df["rot-bad"] == 0]

    # Additional filtering and mutations for demo
    if value_to_predict == "demo":
        df = df.query(
            "`situation-unclear` == 0 & `situation-nsfw` == 0 & `situation-dark` == 0"
        ).query("`rot-char-targeting` == `action-char-involved`")
        da_sel = (df["area"] == "dearabby") & (df["rot-char-targeting"] == "char-0")
        df.loc[da_sel, "rot-char-targeting"] = "char-none"
        df.loc[da_sel, "action-char-involved"] = "char-none"

    num_after = len(df)
    logger.info(
        f"{value_to_predict}-{split}={num_after} (removed {num_before-num_after} instances from with bad RoT)."
    )

    input_type_full = (
        [input_type]
        if input_type != "all"
        else ["situation_and_attributes", "situation"]
    )
    output_type_full = (
        [output_type]
        if output_type != "all"
        else ["action_or_rot", "action_or_rot_and_attributes", "attributes"]
    )

    # Action by rot
    if value_to_predict == "action" and input_type == ["situation_rot_and_attributes"]:
        examples = get_action_from_rot_examples(df)
    elif value_to_predict == "rot":
        examples = get_rot_examples(df, input_type_full, output_type_full)
    elif value_to_predict == "action":
        examples = get_action_examples(df, input_type_full, output_type_full)
    elif value_to_predict == "demo":
        examples = get_demo_examples(df)
    else:
        raise ValueError("value_to_predict must be rot or action")

    return examples


def get_attr(row: pd.Series, col: str, surround: bool = False) -> List[str]:
    res: List[str] = []

    if col == "rot-char-targeting":
        if pd.notna(row["rot-char-targeting"]):
            char_idx_s = row["rot-char-targeting"].split("-")[1]
            if char_idx_s == "none":
                res.append("nobody")
            else:
                char_idx = int(char_idx_s)
                if char_idx > row["n-characters"]:
                    logger.warn(
                        f"Char idx {char_idx} > n chars {row['n-characters']}. Skipping."
                    )
                else:
                    res.append(row["characters"].split("|")[char_idx])
    elif col == "rot-categorization":
        if pd.notna(row["rot-categorization"]):
            # Multi-label
            for category in sorted(row["rot-categorization"].split("|")):
                res.append(f"<{category}>")
    elif col == "rot-moral-foundations":
        if pd.notna(row["rot-moral-foundations"]):
            # Multi-label
            for category in sorted(row["rot-moral-foundations"].split("|")):
                res.append(f"<{category}>")
    elif col == "rot-agree":
        if pd.notna(row["rot-agree"]):
            res.append(f"<{AGREE_TO_STR[row['rot-agree']]}>")
    elif col == "action-agency":
        if pd.notna(row["action-agency"]):
            res.append(f"<{row['action-agency']}>")
    elif col == "action-moral-judgment":
        if pd.notna(row["action-moral-judgment"]):
            res.append(
                f"<{ACTION_MORAL_JUDGMENT_TO_STR[row['action-moral-judgment']]}>"
            )
    elif col == "action-agree":
        if pd.notna(row["action-agree"]):
            res.append(f"<{AGREE_TO_STR[row['action-agree']]}>")
    elif col == "action-legal":
        if pd.notna(row["action-legal"]):
            res.append(f"<{row['action-legal']}>")
    elif col == "action-pressure":
        if pd.notna(row["action-pressure"]):
            res.append(f"<{ACTION_PRESSURE_TO_STR[row['action-pressure']]}>")
    elif col == "action-hypothetical":
        if pd.notna(row["action-hypothetical"]):
            res.append(f"<{row['action-hypothetical']}>")
    else:
        raise ValueError(f"Unknown attribute: '{col}'")

    if len(res) > 0 and surround:
        res = [f"[{col}]"] + res + [f"[/{col}]"]
    return res


def get_demo_special_tokens() -> List[str]:
    """Returns special tokens we add for the demo data processing.

    These include situation, action, judgment, and all attribute delimiters."""
    keys = ["situation", "rot", "action", "judgment"] + DEMO_ATTR_ORDER
    return sorted(flatten([[f"[{k}]", f"[/{k}]"] for k in keys]))


def get_demo_examples(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Gets examples for demo.

    One key feature of the demo data format is that attributes end up on both sides of
    the input & output. This is in contrast to all other setups where attributes are
    fixed on one side. We do this so that 0-n attributes can be specified, and the rest
    will be predicted, so that all attributes end up with values.

    Another is that we mix RoT and action attributes.

    And a third is that we predict all text fields (RoT, action, judgment string).
    """
    # settings; can hook up to be configurable if desired
    attr_drop = 0.1  # in [0, 1]; probability each attribute is dropped
    surround = True  # whether to add <attr-name> </attr-name> tokens around attrs

    # we can't use the get_rot_attributes(...) function because it doesn't group the
    # multi-label attributes. we'd have to modify it. since we need to sort fields and
    # wrap them anyway, we'd might as well do it ourselves.

    # always input:
    # - situation (str)
    # always output:
    # - rot (str)
    # - action (str)
    # - orig judgment (str)

    # generate random states:
    # - inclusion (whether to drop)
    # - set (input or output)
    include = np.random.uniform(size=(len(df), len(DEMO_ATTR_ORDER))) > attr_drop
    sets = np.random.randint(2, size=(len(df), len(DEMO_ATTR_ORDER)))

    # build
    examples = []
    for row_idx, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
        # build up attr reps
        attrs: List[List[str]] = [[], []]  # input, output
        for attr_idx, attr in enumerate(DEMO_ATTR_ORDER):
            if not include[row_idx, attr_idx]:
                continue
            attrs[sets[row_idx, attr_idx]].extend(get_attr(row, attr, surround))

        examples.append(
            (
                f"[situation] {row['situation']} [/situation] {' '.join(attrs[0])}",
                f"{' '.join(attrs[1])} [rot] {row['rot']} [/rot] [action] {row['action']} [/action] [judgment] {row['rot-judgment']} [/judgment] <eos>",
            )
        )

    return examples


def get_rot_examples(df, input_types, output_types):
    """
    Loads the dataset file in the format for training the RoT generative model.

    For instance, for input_types = ["situation_and_attributes"] and output_types = ["action_or_rot"]

    Input: <situation> [attrs] <rot-categorization> <rot-moral-foundations> <rot-agree>
    Output: [rot] <rot>

    df: pandas dataframe
    input_types: list of input types
    output_types: list of output types
    """
    examples = []

    configs = set(list(itertools.product(input_types, output_types)))

    # Predict the rot from the situation and attributes
    if ("situation_and_attributes", "action_or_rot") in configs:
        examples += [
            (
                f"{row['situation']} [attrs] {' '.join(get_rot_attributes(row))} [rot]",
                f"{row['rot']} <eos>" if "rot" in row else None,
            )
            for _, row in df.iterrows()
        ]

    # Predict the rot from the situation
    if ("situation", "action_or_rot") in configs:
        examples += [
            (
                f"{row['situation']} [rot]",
                f"{row['rot']} <eos>" if "rot" in row else None,
            )
            for _, row in df.iterrows()
        ]

    # Predict the attributes from the situation and rot
    if ("situation_and_rot_or_action", "attributes") in configs:
        examples += [
            (
                f"{row['situation']} [rot] {row['rot']} [attrs]",
                f"{' '.join(get_rot_attributes(row))} <eos>",
            )
            for _, row in df.iterrows()
        ]

    # Predict the rot and attributes from the situation
    if ("situation", "action_or_rot_and_attributes") in configs:
        examples += [
            (
                f"{row['situation']} [rot_and_attrs]",
                f"{row['rot']} {' '.join(get_rot_attributes(row))} <eos>",
            )
            for _, row in df.iterrows()
        ]

    return examples


def get_action_from_rot_examples(df):
    """
    Prediction action and action attributes given the RoT and RoT attributes
    """
    examples = [
        (
            f"{row['situation']} [rot] {row['rot']} [attrs] {' '.join(get_rot_attributes(row))} [action_and_attrs]",
            f"{row['action']} {' '.join(get_action_attributes(row))} <eos>",
        )
        for _, row in df.iterrows()
    ]

    return examples


def get_action_examples(df, input_types, output_types):
    """
    Loads the dataset file in the format for training the action generative model.

    For instance, for input_types = ["situation_and_attributes"] and output_types = ["action_or_rot"]

    <situation> [attrs] <action-agency> <action-moral-judgment> <action-agree>
    <action-legal> <action-pressure> <action-hypothetical> [action]
    Output: <action>

    df: pandas dataframe
    input_types: list of input types
    output_types: list of output types
    """
    examples = []

    configs = set(list(itertools.product(input_types, output_types)))

    # Predict the action from the situation and attributes
    if ("situation_and_attributes", "action_or_rot") in configs:
        examples += [
            (
                f"{row['situation']} [attrs] {' '.join(get_action_attributes(row))} [action]",
                f"{row['action']} <eos>" if "action" in row else None,
            )
            for _, row in df.iterrows()
        ]

    # Predict the action from the situation
    if ("situation", "action_or_rot") in configs:
        examples += [
            (
                f"{row['situation']} [action]",
                f"{row['action']} <eos>" if "action" in row else None,
            )
            for _, row in df.iterrows()
        ]

    # Predict the attributes from the situation and action
    if ("situation_and_rot_or_action", "attributes") in configs:
        examples += [
            (
                f"{row['situation']} [action] {row['action']} [attrs]",
                f"{' '.join(get_action_attributes(row))} <eos>",
            )
            for _, row in df.iterrows()
        ]

    # Predict the action and attributes from the situation
    if ("situation", "action_or_rot_and_attributes") in configs:
        examples += [
            (
                f"{row['situation']} [action_and_attrs]",
                f"{row['action']} {' '.join(get_action_attributes(row))} <eos>",
            )
            for _, row in df.iterrows()
        ]

    return examples


def display_input_lengths(c: typing.Counter[int]) -> None:
    max_len = max(c.keys())
    total = sum(c.values())

    # this would probably be faster with numpy
    def fit_report(candidate: int) -> None:
        under, over = 0, 0
        for l, cnt in c.items():
            if l <= candidate:
                under += cnt
            else:
                over += cnt
        logger.info(
            f" - if max len {candidate}, {under}/{total} fit ({100*under/total:.1f}%)"
        )

    candidate = 1
    logger.info("Max len analysis:")
    logger.info(f"{total} items, maximum observed length {max_len}")
    while candidate < max_len:
        fit_report(candidate)
        candidate *= 2
    fit_report(candidate)
