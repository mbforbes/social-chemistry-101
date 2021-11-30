"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import re
import json
import tqdm
import torch
import logging
import argparse
import pandas as pd


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

from sc.model.common import init_model, load_data, get_all_attributes


def main() -> None:
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--in_file",
        default=None,
        type=str,
        required=True,
        help="in TSV file of Social Chemistry 101 data",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        required=True,
        help="out jsonl file with RoTs",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
    parser.add_argument(
        "--value_to_predict",
        default="rot",
        type=str,
        help="Predict RoT (rot) or action.",
    )
    parser.add_argument(
        "--input_type",
        default="situation_and_attributes",
        type=str,
        help="situation_and_attributes | situation | all (multi-task)",
    )
    parser.add_argument(
        "--split",
        default=None,
        type=str,
        help="which split to load from the data (None = load everything).",
    )
    parser.add_argument(
        "--output_type",
        default="action_or_rot",
        type=str,
        help="action_or_rot | action_or_rot_and_attributes | attributes | all (multi-task)",
    )
    parser.add_argument(
        "--max_length", default=40, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    args = parser.parse_args()
    logger.debug(args)

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    args.device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, args)

    examples = load_data(
        args.in_file,
        args.value_to_predict,
        input_type=args.input_type,
        output_type=args.output_type,
        split=args.split,
    )
    print(examples[:10])

    attributes = get_all_attributes(pd.read_csv(args.in_file, delimiter="\t"))
    special_tokens = [
        "[attrs]",
        "[rot]",
        "[action]",
        "[rot_and_attrs]",
        "[action_and_attrs]",
    ]

    generate = (
        generate_conditional
        if "t5" in args.model_name_or_path or "bart" in args.model_name_or_path
        else generate_regular
    )

    with open(args.out_file, "w") as f_out:
        for input, output in tqdm.tqdm(examples):
            try:
                skip_attributes = not input.endswith("attrs]")
                pred = generate(
                    tokenizer,
                    model,
                    args,
                    input,
                    args.device,
                    skip_attributes,
                )

                if skip_attributes:
                    for special_token in attributes:
                        pred = pred.replace(special_token, "")

                for special_token in special_tokens:
                    pred = pred.replace(special_token, "")

                pred = re.sub(" +", " ", pred).strip()

                # TODO: remove the entire part until after “] ” in cases in which the required output is a free text.
                if pred.startswith("] "):
                    pred = pred[2:]

            except Exception as exp:
                logger.info(exp)
                pred = ""

            f_out.write(
                json.dumps({"input": input, "output": output, "prediction": pred})
                + "\n"
            )


def generate_conditional(
    tokenizer,
    model,
    args,
    input,
    device,
    skip_attributes,
):
    """
    Generate a sequence with models like Bart and T5
    """
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    decoder_start_token_id = input_ids[-1]
    input_ids = torch.tensor([input_ids]).to(device)
    max_length = args.max_length

    outputs = model.generate(
        input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        min_length=5,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
    )

    pred = tokenizer.decode(
        outputs[0],
        skip_special_tokens=skip_attributes,
        clean_up_tokenization_spaces=False,
    )

    return pred


def generate_regular(
    tokenizer,
    model,
    args,
    input,
    device,
    skip_attributes,
):
    """
    Generate a sequence with models like GPT, GPT2, or XLNet
    """
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    input_length = len(input_ids)
    max_length = args.max_length + input_length
    input_ids = torch.tensor([input_ids]).to(device)

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
    )

    pred = tokenizer.decode(
        outputs[0],
        skip_special_tokens=skip_attributes,
        # clean_up_tokenization_spaces=False,
    )[len(input) :]

    return pred


if __name__ == "__main__":
    main()
