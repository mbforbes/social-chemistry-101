"""Manually cleaning some model outputs that have glitches in the
output/postprocessing.

Usage:
python -m sc.model.cleaning \
    --input-path data/output/generate/rot_all_gpt2random-xl/test_predictions_p0.9.jsonl \
    --output-path data/output/generate/rot_all_gpt2random-xl/test_predictions_p0.9.cleaned.jsonl
"""

import argparse
import code
import json
import logging
import re
from re import Pattern  # type: ignore
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator

from mbforbes_python_utils import read, write
from tqdm import tqdm

RE_L_POINTY = re.compile("<")
RE_R_POINTY = re.compile(">")
RE_L_SQUARE = re.compile("\[")
RE_R_SQUARE = re.compile("\]")


def clean(s: str, r_char: str) -> str:
    idx = s.find(r_char)

    # sanity check heuristic that it really is at ~beginning of string
    if idx > 15:
        raise ValueError(f"'{r_char}' found too far ({idx}) in '{s}'")

    # there's a space after the brackets in every case i've seen. for convenience,
    # remove those as well.
    new_start = idx + 1
    if idx + 1 < len(s) and s[idx + 1] == " ":
        new_start = idx + 2

    return s[new_start:]


def maybe_clean(s: str, r_char: str, re_l: Pattern, re_r: Pattern) -> str:
    n_l = len(re_l.findall(s))
    n_r = len(re_r.findall(s))
    if n_l == n_r:
        # good
        return s
    elif n_l + 1 == n_r:
        # case we've seen: one too many r chars.
        return clean(s, r_char)
    else:
        raise ValueError(f"Unhandled mismatch! {n_l}/{n_r} ({r_char}): {s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to model output file(s) (jsonl file = one json object per line).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to model output file(s) (jsonl file = one json object per line).",
    )
    args = parser.parse_args()

    lines = [json.loads(line.strip()) for line in read(args.input_path).split("\n")]
    cleaned = 0
    for line in tqdm(lines):
        for field in ["input", "output", "prediction"]:
            s = line[field]
            s = maybe_clean(s, ">", RE_L_POINTY, RE_R_POINTY)
            s = maybe_clean(s, "]", RE_L_SQUARE, RE_R_SQUARE)
            # Slow, but reporting is nice.
            if s != line[field]:
                cleaned += 1
            line[field] = s
    print(f"Cleaned {cleaned} lines.")
    if cleaned == 0:
        print("Not writing new file because nothing changed.")
        return
    write(args.output_path, "\n".join([json.dumps(line) for line in lines]) + "\n")


if __name__ == "__main__":
    main()
