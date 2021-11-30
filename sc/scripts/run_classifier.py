"""Runs a trained classifier on model output.

Usage:

python -m sc.scripts.run_classifier

This is a script runner that calls sc/model/classify.py under the hood. Examples of the
kinds of commands it will run:

# eval the dataset
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path checkpoints/rot-agree/roberta-base/ \
    --attr rot-agree \
    --input_situation \
    --input_rot \
    --do_eval \
    --output_dir data/output/classify/

# eval model outputs
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path checkpoints/rot-agree/roberta-base/ \
    --attr rot-agree \
    --input_situation \
    --input_rot \
    --do_eval \
    --data_dir data/output/classify/input/test_bart-large_rot_controlled/ \
    --data_filename data.tsv \
    --output_dir data/output/classify/input/test_bart-large_rot_controlled/

columns the classifier needs:                                                  vals
- [x] characters <---------- NOT DOING            n/a
- [x] rot-bad               (as preprocessing)    0
- [x] split                 (as preprocessing)    test
- [x] rot-id                (as guid)             anything unique
- [x] situation             (as input)            legit
- [x] rot                   (as input)            rot-only
- [x] action                (as input)            action-only
- [x] rot-agree             (as label)            rot-only
- [x] rot-moral-foundations (as label)            rot-only
- [x] rot-categorization    (as label)            rot-only
- [x] action-agency         (as label)            action-only
- [x] action-moral-judgment (as label)            action-only
- [x] action-agree          (as label)            action-only
- [x] action-legal          (as label)            action-only
- [x] action-pressure       (as label)            action-only
- [x] action-hypothetical   (as label)            action-only


# sit rot act jmt chr qst attr
# --- --- --- --- --- --- ---
#  x   x              C   rot-agree
#  x   x              ML  rot-moral-foundations
#  x   x              ML  rot-categorization
#  x       x          C   action-agency
#  x       x          C   action-moral-judgment (rate)
#          x   x      C   action-agree
#  x       x          C   action-legal
#  x       x          C   action-pressure
#  x       x          C   action-hypothetical

"""

import code
import glob
import json
import os
import subprocess

from mbforbes_python_utils import read

# settings

action_attrs = [
    "action-agency",
    "action-moral-judgment",
    "action-agree",
    "action-legal",
    "action-pressure",
    "action-hypothetical",
]

rot_attrs = [
    "rot-agree",
    "rot-moral-foundations",
    "rot-categorization",
]

model_type = "roberta"
model_name_or_path = "checkpoints/"


def has_run(d: str, attr: str) -> bool:
    eval_path = os.path.join(d, "eval_results.jsonl")
    if not os.path.exists(eval_path):
        return False
    results = [json.loads(line) for line in read(eval_path).split("\n")]
    for result in results:
        if result["attr"] == attr:
            return True
    return False


def main() -> None:
    dirs = glob.glob("data/output/classify/input/*")
    already_run, total = 0, 0
    for d in dirs:
        # print(d)
        generative_setup = os.path.basename(d)
        mode = generative_setup.split("_")[2]
        assert mode in {"rot", "action"}

        attr_list = rot_attrs if mode == "rot" else action_attrs
        for attr in attr_list:
            # check whether we need to run
            total += 1
            if has_run(d, attr):
                already_run += 1
                continue

            if mode == "rot":
                input1, input2 = "--input_situation", "--input_rot"
            elif attr == "action-agree":
                input1, input2 = "--input_action", "--input_judgment"
            else:
                input1, input2 = "--input_situation", "--input_action"

            checkpoint = f"{attr}-rate" if attr == "action-moral-judgment" else attr

            args = [
                "python",
                "-m",
                "sc.model.classify",
                "--model_type",
                "roberta",
                "--model_name_or_path",
                f"checkpoints/{checkpoint}/roberta-base/",
                "--attr",
                f"{attr}",
                f"{input1}",
                f"{input2}",
                "--do_eval",
                "--data_dir",
                f"{d}",
                "--data_filename",
                "data.tsv",
                "--output_dir",
                f"{d}",
            ]
            subprocess.run(args)

    # print(f"Already ran {already_run}/{total} configs; {total-already_run} remain")


if __name__ == "__main__":
    main()
