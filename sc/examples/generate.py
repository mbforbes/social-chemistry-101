"""Example of generating with model.

Dynamically loads situations and axes of variation found in sc/examples/examples.py
and generates from those constraints using specified model.

Generation code originally based off of:
    https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
... which is now at (and has probably changed; check git history to find orig.):
    https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py

Usage (download or train models first):

# RoTs:
python -m sc.examples.generate --model output/gpt2-xl_rot_64_5epochs

# Actions:
python -m sc.examples.generate --model output/gpt2-xl_action_64_5epochs
"""


import argparse
import code
from importlib import reload
import json
import logging
import re
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARN,
)

logger = logging.getLogger(__name__)

import pandas as pd
import torch
import tqdm

from sc.model.common import init_model, load_data, get_all_attributes
from sc.examples import examples


def main() -> None:
    """
    Generate ea RoT for each situation the text
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--model",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
    parser.add_argument(
        "--max_length", default=40, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--do_lower_case", default=True, type=bool, required=False, help="lower?"
    )
    # Using p=0.9 by default
    parser.add_argument(
        "--p", default=0.9, type=float, required=False, help="p for nucleus sampling"
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
    parser.add_argument("--device", default="0", type=str, help="GPU number or 'cpu'.")
    args = parser.parse_args()

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    args.device = device

    mode = 'action' if 'action' in args.model else 'rot'

    tokenizer, model = init_model(args.model, args)
    eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]

    # just caching these for now
    special_tokens = [
        "<controversial>",
        "<against>",
        "<for>",
        "<description>",
        "<legal>",
        "<care-harm>",
        "<explicit-no>",
        "<loyalty-betrayal>",
        "<explicit>",
        "<all>",
        "<strong-against>",
        "<morality-ethics>",
        "<good>",
        "<probable>",
        "<social-norms>",
        "<authority-subversion>",
        "<most>",
        "<ok>",
        "<hypothetical>",
        "<discretionary>",
        "<fairness-cheating>",
        "<bad>",
        "<agency>",
        "[attrs]",
        "[rot]",
        "[action]",
        "<eos>",
    ]

    while True:
        print("Loading examples from sc/examples/examples.py ...")
        ex = examples.rot_examples() if mode == 'rot' else examples.action_examples()
        print()

        for prompt, (variable, options) in ex:
            print()
            print(f'Situation: "{prompt}"')
            print(f"-- Varying along '{variable}' --")
            n_gens = 5
            for option in options:
                input_ = prompt.format(varies=option)
                input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_))
                input_length = len(input_ids)
                max_length = args.max_length + input_length
                input_ids = torch.tensor([input_ids]).to(device)

                # NOTE: swap this out for newer model
                eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]
                # eos_token_id = tokenizer.eos_token_id,

                print(f"\t{option}")
                for _ in range(n_gens):

                    outputs = model.generate(
                        input_ids,
                        do_sample=args.beams == 0,
                        max_length=max_length,
                        temperature=args.temperature,
                        top_p=args.p if args.p > 0 else None,
                        top_k=args.k if args.k > 0 else None,
                        eos_token_id=eos_token_id,
                        num_beams=args.beams if args.beams > 0 else None,
                        early_stopping=True,
                        # NOTE: swap for new models that (I think) actually have pad id set.
                        # pad_token_id=tokenizer.pad_token_id,
                        pad_token_id=50256,
                        no_repeat_ngram_size=3,
                    )

                    pred = tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                        # clean_up_tokenization_spaces=False,
                    )[len(input_):].strip()

                    # Include this if, for some reason, any special tokens remain
                    # for special_token in special_tokens:
                    #     pred = pred.replace(special_token, "")
                    # pred = re.sub(" +", " ", pred)

                    print(f"\t\t{pred}")

        print()
        if input("[q=quit, anything else=reload and generate]> ") == 'q':
            return
        reload(examples)


if __name__ == "__main__":
    main()
