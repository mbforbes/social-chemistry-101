"""Common mturk utils."""


import code
import json
import os
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator
from typing_extensions import TypedDict, Literal, Final

from mbforbes_python_utils import read


## types ##

# Can't use the constants in Literals 😢
Mode = Literal["rot", "action"]
Task = Literal["model_choice", "controlled"]


class OutputInfo(TypedDict):
    model_name: str
    model_size: str
    training_scheme: str
    mode: Mode
    split: str
    sampling: str
    short_path: str


class OutputLine(TypedDict):
    input: str
    output: str
    prediction: str


class ModelOutput(TypedDict):
    info: OutputInfo
    lines: List[OutputLine]


## globals ##

M_ROT: Final = "rot"
M_ACTION: Final = "action"


## funcs ##


def get_model_output(path: str, debug_print: bool = True) -> ModelOutput:
    """Makes STRONG assumptions on path and filename to get info.

    Expects them to look like one of:
        .../action_all_bart-large/test_predictions_p0.9.jsonl
        .../action_all_gpt2/dev_predictions_p0.9.jsonl

    ... with either of those set of components as delineated by - _ .

    TODO: migrate this to sc/model/common.py or something
    """
    d = os.path.basename(os.path.dirname(path))
    f = os.path.basename(path)

    mode, training_scheme, full_model = d.split("_")
    model_size = "default"
    if "-" in full_model:
        model_name, model_size = full_model.split("-")
    else:
        model_name = full_model
    split, _, sampling = (".".join(f.split(".")[:-1])).split("_")
    info: OutputInfo = dict(
        model_name=model_name,
        model_size=model_size,
        training_scheme=training_scheme,
        mode=mode,  # type: ignore
        split=split,
        sampling=sampling,
        short_path=os.path.join(d, f),
    )

    # sanity check reporting
    if debug_print:
        print(f"Extrated from '{path}':")
        [print(f"- {k}: {v}") for k, v in info.items()]  # type:ignore

    # actually load
    lines = [json.loads(line.strip()) for line in read(path).split("\n")]

    return dict(info=info, lines=lines)
