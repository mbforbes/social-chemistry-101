"""Human eval utils: input."""

import argparse
import code
from collections import Counter
import glob
import json
import logging
import os
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union
from typing_extensions import Final

import pandas as pd
from tqdm import tqdm

from sc.mturk.common import (
    OutputInfo,
    OutputLine,
    ModelOutput,
    Mode,
    Task,
    M_ROT,
    M_ACTION,
    T_MODEL_CHOICE,
    T_CONTROLLED,
)

# TODO: migrate some of these to sc/model/common.py or somewhere
MODES: List[Mode] = [M_ROT, M_ACTION]
TASKS: List[Task] = [T_MODEL_CHOICE, T_CONTROLLED]
SETUPS: Dict[Tuple[Mode, Task], str] = {
    (M_ROT, T_MODEL_CHOICE): "S->RB",
    (M_ROT, T_CONTROLLED): "SB->R",
    (M_ACTION, T_MODEL_CHOICE): "S->AB",
    (M_ACTION, T_CONTROLLED): "SB->A",
}
REV_SETUPS: Dict[str, Tuple[Mode, Task]] = {v: k for k, v in SETUPS.items()}


def get_setup(output_line: OutputLine, mode: Mode) -> str:
    """TODO: migrate to sc/model/common.py or somewhere."""
    i = output_line["input"]
    if mode == "rot":
        if i.endswith("[rot_and_attrs]"):
            return "S->RB"
        elif i.endswith("[rot]"):
            if "[attrs]" in i:
                return "SB->R"
            else:
                return "S->R"
        else:
            raise ValueError(f"Unknown setup for input: {i}")
    elif mode == "action":
        if i.endswith("[action_and_attrs]"):
            return "S->AB"
        elif i.endswith("[action]"):
            if "[attrs]" in i:
                return "SB->A"
            else:
                return "S->A"
        else:
            raise ValueError(f"Unknown setup for input: {i}")
    else:
        raise ValueError("Mode {mode} not yet implemented")


def get_main_item(
    model_output_line: Union[pd.Series, OutputLine], mode: Mode, task: Task
) -> Dict[str, str]:
    """Returns RoT or action output by model, no attrs or special tags."""
    # (M_ROT, T_MODEL_CHOICE): "S->RB",
    # {
    #   "input": "telling my boyfriend I am bored and unhappy at my job [rot_and_attrs]",
    #   "output": "Partners should listen to each other's issues.
    #              <social-norms> <care-harm> <loyalty-betrayal> <most> <eos>",
    #   "prediction": "It is normal to feel frustrated at work sometimes.
    #                  <description> <authority-subversion> <all> <eos>"
    # }
    pred = model_output_line["prediction"]
    if mode == M_ROT and task == T_MODEL_CHOICE:
        first_tag = pred.find("<")
        if first_tag != -1:
            pred = pred[:first_tag]
        return {"rot": pred.strip()}

    # (M_ROT, T_CONTROLLED): "SB->R",
    # {
    #   "input": "telling my boyfriend I am bored and unhappy at my job [attrs]
    #             <social-norms> <care-harm> <loyalty-betrayal> <most> [rot]",
    #   "output": "Partners should listen to each other's issues. <eos>",
    #   "prediction": "Partners are expected to talk about their problems with each
    #                  other."
    # }
    if mode == M_ROT and task == T_CONTROLLED:
        return {"rot": pred}

    # (M_ACTION, T_MODEL_CHOICE): "S->AB",
    # {
    #   "input": "telling my boyfriend I am bored and unhappy at my job [action_and_attrs]",
    #   "output": "Listening to each other's issues. <agency> <very-good> <most> <legal>
    #              <strong-for> <probable> <eos>",
    #   "prediction": "being honest with your partner. <agency> <good> <all> <legal>
    #                  <strong-for> <explicit> <eos>"
    # }
    if mode == M_ACTION and task == T_MODEL_CHOICE:
        first_tag = pred.find("<")
        if first_tag != -1:
            pred = pred[:first_tag]
        return {"action": pred.strip()}

    # (M_ACTION, T_CONTROLLED): "SB->A",
    # {
    #   "input": "telling my boyfriend I am bored and unhappy at my job [attrs] <agency>
    #             <very-good> <most> <legal> <strong-for> <probable> [action]",
    #   "output": "Listening to each other's issues. <eos>",
    #   "prediction": "telling someone you are bored and doing something else."
    # }
    if mode == M_ACTION and task == T_CONTROLLED:
        return {"action": pred}

    raise ValueError(f"Unsupported mode/task combination {mode}/{task}")
