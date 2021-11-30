"""Turns model output into table format that can be used by classifier.

In hindsight, I should have written this before writing a million bidirectional
converters for human eval, but ¯\_(ツ)_/¯

Usage:
    python -m sc.scripts.output_to_table
"""

import argparse
import code
import glob
import logging
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator

import pandas as pd
from tqdm import tqdm

from sc.mturk.common import Task, M_ROT, M_ACTION
from sc.mturk import common as mturk_common
from sc.mturk.input.human_eval import REV_SETUPS, TASKS
from sc.mturk.input import human_eval as input_he
from sc.mturk.output.human_eval import ActionAttrs, RoTAttrs


def get_situation(input_: str) -> str:
    end = input_.find("[") - 1
    res = input_[:end]
    for baddie in ["[", "]", "<", ">"]:
        if baddie in res:
            msg = f"Unexpected tkn '{baddie}' in res '{res}'"
            print(msg)
            code.interact(local=dict(globals(), **locals()))
            raise ValueError(msg)
    return res


def main() -> None:
    out_dir = "data/output/classify/input/"
    paths = glob.glob("data/output/generate/**/test_predictions_p0.9.jsonl")

    for path in tqdm(paths):
        # unconditioned models are like an ablation, and don't fit our formatting
        # expectations.
        if "unconditioned" in path:
            continue

        model_output = mturk_common.get_model_output(path, debug_print=False)
        info = model_output["info"]
        mode, model_name, model_size = (
            info["mode"],
            info["model_name"],
            info["model_size"],
        )
        task_rows: Dict[Task, List[Dict[str, Any]]] = {task: [] for task in TASKS}

        for idx, line in enumerate(model_output["lines"]):
            setup = input_he.get_setup(line, mode)
            if setup in {"S->R", "S->A"}:
                continue
            _, task = REV_SETUPS[setup]
            row_common: Dict[str, Any] = {
                "rot-bad": 0,
                "split": "test",
                "rot-id": f"rot/test/{model_name}-{model_size}/{mode}/{task}/{idx}",
                "situation": get_situation(line["input"]),
            }
            if mode == M_ROT:
                rot_attrs = RoTAttrs.from_model(line, task)
                row: Dict[str, Any] = {
                    **row_common,
                    "rot": input_he.get_main_item(line, mode, task)["rot"],
                    **rot_attrs.to_row_builder(),
                }
            elif mode == M_ACTION:
                action_attrs = ActionAttrs.from_model(line, task)
                row: Dict[str, Any] = {  # type: ignore
                    **row_common,
                    "action": input_he.get_main_item(line, mode, task)["action"],
                    **action_attrs.to_row_builder(),
                }
            else:
                raise ValueError(f"Unsupported {mode}")

            task_rows[task].append(row)

        for task, rows in task_rows.items():
            out_path = out_dir + f"test_{model_name}-{model_size}_{mode}_{task}.tsv"
            # print(f"Writing to {out_path}")
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
