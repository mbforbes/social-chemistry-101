"""Human eval utils: output."""

import argparse
import code
from collections import Counter
import copy
import glob
import logging
import os
import re
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union
from typing_extensions import TypedDict, Literal

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

from sc.mturk.common import (
    OutputInfo,
    OutputLine,
    Mode,
    Task,
    M_ROT,
    M_ACTION,
    T_MODEL_CHOICE,
    T_CONTROLLED,
    TASKS,
)
from sc.mturk import common as mturk_common
from sc.model.common import (
    STR_TO_AGREE,
    STR_TO_ACTION_MORAL_JUDGMENT,
    STR_TO_ACTION_PRESSURE,
)


## types & util classes ##

TAG_FINDER = re.compile("<[^>]*>")
# done
EVAL_HEADERS = ["coherent", "generic", "relevant"]

CAT_COLS = ["cat-morality-ethics", "cat-social-norms", "cat-advice", "cat-description"]
MF_COLS = [
    "mf-care-harm",
    "mf-fairness-cheating",
    "mf-loyalty-betrayal",
    "mf-authority-subversion",
    "mf-sanctity-degradation",
]
ROT_SIMPLE_ATTR_COLS = ["rot-agree"]
ROT_AGG_ATTR_COLS = ["cat", "mf"]
ACTION_ATTR_COLS = [
    "action-agency",
    "action-moral-judgment",
    "action-agree",
    "action-pressure",
    "action-legal",
    "action-hypothetical",
]
OVERALL_ATTR_COL = "overall-attr"
OVERALL_EVAL_COL = "overall-eval"

# attrs for which we have labels as strings
STR_LABEL_ATTRS = {"action-agency", "action-legal", "action-hypothetical"}


class ScoreData(TypedDict):
    pred: List[Any]
    labels: List[List[Any]]


class MetricSet(TypedDict):
    precision: float
    recall: float
    f1_micro: float
    f1_macro: float
    accuracy: float
    human_avg: float


MetricName = Literal[
    "precision", "recall", "f1_micro", "f1_macro", "accuracy", "human_avg"
]


class Scores(object):
    def __init__(self) -> None:
        """Add scores and ratings, and only once you're finished adding, should you
        get metrics, because results are cached permanently and never updated."""

        self.scores: Dict[Task, Dict[str, ScoreData]] = {}
        self.ratings: Dict[Task, Dict[str, List[float]]] = {}

        self.metric_cache: Dict[Task, Dict[str, MetricSet]] = {
            task: {} for task in TASKS
        }
        self.rating_cache: Dict[Task, Dict[str, float]] = {task: {} for task in TASKS}

    def add_rating(self, task: Task, key: str, rating: float) -> None:
        if task not in self.ratings:
            self.ratings[task] = {}
        if key not in self.ratings[task]:
            self.ratings[task][key] = []
        self.ratings[task][key].append(rating)

    def add(self, task: Task, key: str, pred: Any, labels: List[Any]) -> None:
        if task not in self.scores:
            self.scores[task] = {}
        if key not in self.scores[task]:
            self.scores[task][key] = {"pred": [], "labels": []}
        self.scores[task][key]["pred"].append(pred)
        self.scores[task][key]["labels"].append(labels)

    def _get_metrics(self, task: Task, key: str) -> MetricSet:
        """Returns P, R, F1-micro, F1-macro, accuracy, human avg score."""
        if key in self.metric_cache[task]:
            return self.metric_cache[task][key]

        pred, labels = self.scores[task][key]["pred"], self.scores[task][key]["labels"]

        # first, compute human avg.
        human_scores = []
        for p, ll in zip(pred, labels):
            human_scores.append(sum([p == l for l in ll]) / len(ll))
        human_avg = np.array(human_scores).mean()

        # now, compute "actual" labels from mturk labels, and use to calculate standard
        # metrics.
        actual = []
        for ll in labels:
            # get most common label. for >= 2/3, we use it. otherwise, we say invalid.
            assert len(ll) == 3  # if this isn't true, we can change the numbers
            label, count = Counter(ll).most_common(1)[0]
            if count == 1:
                label = "NO_AGREE" if key in STR_LABEL_ATTRS else -100
            actual.append(label)
        assert len(actual) == len(pred)

        p, r, f1_micro, _ = metrics.precision_recall_fscore_support(
            y_true=actual, y_pred=pred, average="micro"
        )
        f1_macro = metrics.f1_score(y_true=actual, y_pred=pred, average="macro")
        acc = sum([a == p for a, p in zip(actual, pred)]) / len(actual)

        return {
            "precision": p,
            "recall": r,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "accuracy": acc,
            "human_avg": human_avg,
        }

    def _get_rating(self, task: Task, key: str) -> float:
        """Returns avg rating for this key."""
        if key in self.rating_cache[task]:
            return self.rating_cache[task][key]
        avg = np.array(self.ratings[task][key]).mean()
        self.rating_cache[task][key] = avg
        return avg

    def _aggregate(self, metric_sets: List[MetricSet], metric: MetricName) -> float:
        """Handle correctly (hopefully) aggregating multiple metric sets."""
        # Things should be simplified because all of the metrics will be for the same
        # number of examples.

        # We're going to average macro F1 scores. This should be the same, since the
        # relative proportions of the sets were already averaged into the individual
        # macro F1 scores, and each set of metrics that will be combined has the same
        # support because they are the same size (e.g., 200 examples).

        # These we can just average, I think.
        if metric in {"precision", "recall", "accuracy", "human_avg", "f1_macro"}:
            return np.array([m[metric] for m in metric_sets]).mean()

        # We're going to compute "overall F1" as the f1_micro.
        if metric == "f1_micro":
            p = np.array([m["precision"] for m in metric_sets]).mean()
            r = np.array([m["recall"] for m in metric_sets]).mean()
            return (2 * p * r) / (p + r)

        raise ValueError(f"Unknown metric: {metric}")

    def get(self, mode: Mode, task: Task, col: str, metric: MetricName) -> float:
        """
        Notes:

            - mode: each Score is already fixed to a Mode; it's just so we know which
            columns to aggregate over.

            - metric: ratings ignore metric; they just use avg.
        """
        # ratings -- ignore metric
        if col in EVAL_HEADERS:
            return self._get_rating(task, col)
        if col == OVERALL_EVAL_COL:
            return np.array([self._get_rating(task, c) for c in EVAL_HEADERS]).mean()

        # attrs that don't require aggregation
        if col in (CAT_COLS + MF_COLS + ROT_SIMPLE_ATTR_COLS + ACTION_ATTR_COLS):
            return self._get_metrics(task, col)[metric]

        # require aggregating
        if col in ROT_AGG_ATTR_COLS:
            if col == "cat":
                return self._aggregate(
                    [self._get_metrics(task, c) for c in CAT_COLS], metric
                )
            elif col == "mf":
                return self._aggregate(
                    [self._get_metrics(task, c) for c in MF_COLS], metric
                )
            else:
                raise ValueError(f"Unknown ROT AGG ATTR COL {col}")

        if col == OVERALL_ATTR_COL:
            if mode == M_ACTION:
                return self._aggregate(
                    [self._get_metrics(task, c) for c in ACTION_ATTR_COLS], metric
                )
            elif mode == M_ROT:
                cols = ROT_SIMPLE_ATTR_COLS + ROT_AGG_ATTR_COLS
                if metric == "f1_micro":
                    # do an "overall" f1 score for this
                    p = np.array(
                        [self.get(mode, task, c, "precision") for c in cols]
                    ).mean()
                    r = np.array(
                        [self.get(mode, task, c, "recall") for c in cols]
                    ).mean()
                    return (2 * p * r) / (p + r)
                else:
                    return np.array(
                        [self.get(mode, task, c, metric) for c in cols]
                    ).mean()
            else:
                raise ValueError(f"Unknown mode {mode}")

        raise ValueError(f"Unknown col {col}")


AttrDict = Dict[str, Union[str, int]]


def attr_from_str(
    base: AttrDict, tag_map: Dict[str, Tuple[str, Union[str, int]]], s: str, mode: Mode
) -> Dict[str, Union[str, int]]:
    res = copy.deepcopy(base)
    for tag_raw in TAG_FINDER.findall(s):
        tag = tag_raw[1:-1]
        if tag == "eos":
            pass
        elif tag in tag_map:
            k, v = tag_map[tag]
            res[k] = v
        else:
            # print(f"WARNING: Unrecognized {mode} tag {tag_raw}")
            pass
    return res


def score_attrs(
    scores: Scores, task: Task, gold_attrs: AttrDict, mturk_attrs: List[AttrDict]
) -> None:
    for key, gold_val in gold_attrs.items():
        scores.add(task, key, gold_val, [r[key] for r in mturk_attrs])


class ActionAttrs(object):

    # build map from {<tag>: (attr, value)}
    tag_map: Dict[str, Tuple[str, Union[str, int]]] = {}
    for tag in ["agency", "experience"]:
        tag_map[tag] = ("action-agency", tag)
    for tag in ["illegal", "tolerated", "legal"]:
        tag_map[tag] = ("action-legal", tag)
    for s, n in STR_TO_ACTION_MORAL_JUDGMENT.items():
        tag_map[s] = ("action-moral-judgment", n)
    for s, n in STR_TO_AGREE.items():
        tag_map[s] = ("action-agree", n)
    for s, n in STR_TO_ACTION_PRESSURE.items():
        tag_map[s] = ("action-pressure", n)
    for tag in ["explicit", "explicit-no", "hypothetical", "probable", "probable-no"]:
        tag_map[tag] = ("action-hypothetical", tag)

    base: AttrDict = {
        "action-agency": "",
        "action-legal": "",
        "action-moral-judgment": -100,
        "action-agree": -100,
        "action-pressure": -100,
        "action-hypothetical": "",
    }

    def __init__(self, attrs: AttrDict) -> None:
        self.attrs = attrs

    def to_row_builder(self) -> Dict[str, Any]:
        """Puts attrs in pd row-friendly dict in dataset format."""
        return {
            "action-agency": self.attrs["action-agency"],
            "action-moral-judgment": self.attrs["action-moral-judgment"],
            "action-agree": self.attrs["action-agree"],
            "action-legal": self.attrs["action-legal"],
            "action-pressure": self.attrs["action-pressure"],
            "action-hypothetical": self.attrs["action-hypothetical"],
        }

    @staticmethod
    def from_str(s: str) -> "ActionAttrs":
        return ActionAttrs(
            attr_from_str(ActionAttrs.base, ActionAttrs.tag_map, s, M_ACTION)
        )

    @staticmethod
    def from_mturk(row: pd.Series) -> "ActionAttrs":
        res = copy.deepcopy(ActionAttrs.base)

        res["action-agency"] = row["Answer.agency-fwd"]
        res["action-legal"] = row["Answer.fwd-legal"]

        mturk_keys = [
            "Answer.action-fwd-good",
            "Answer.fwd-agree",
            "Answer.fwd-pressure",
        ]
        attr_keys = ["action-moral-judgment", "action-agree", "action-pressure"]
        for mturk_key, attr_key in zip(mturk_keys, attr_keys):
            if not pd.isna(row[mturk_key]) and row[mturk_key] != "":
                res[attr_key] = int(row[mturk_key])

        res["action-hypothetical"] = row["Answer.fwd-hypothetical"]

        return ActionAttrs(res)

    @staticmethod
    def from_model(line: OutputLine, task: Task) -> "ActionAttrs":
        if task == T_MODEL_CHOICE:
            # model chooses, so it was in the prediction
            return ActionAttrs.from_str(line["prediction"])
        elif task == T_CONTROLLED:
            # controlled, so it was in the input
            return ActionAttrs.from_str(line["input"])
        else:
            raise ValueError(f"Unknown task {task}")

    @staticmethod
    def score(
        scores: Scores, task: Task, gold: "ActionAttrs", mturk: List["ActionAttrs"]
    ) -> None:
        score_attrs(scores, task, gold.attrs, [m.attrs for m in mturk])


class RoTAttrs(object):

    # Yep this is a bunch of code in the class body itself. :dealwithitparrot:

    # build up the mapping. given a tag (which are thankfully globally unique among RoTs
    # and actions separately (though not together), get back the attribute and the value
    # it represents.
    tag_map: Dict[str, Tuple[str, Union[str, int]]] = {
        "morality-ethics": ("cat-morality-ethics", 1),
        "social-norms": ("cat-social-norms", 1),
        "advice": ("cat-advice", 1),
        "description": ("cat-description", 1),
        "care-harm": ("mf-care-harm", 1),
        "fairness-cheating": ("mf-fairness-cheating", 1),
        "loyalty-betrayal": ("mf-loyalty-betrayal", 1),
        "authority-subversion": ("mf-authority-subversion", 1),
        "sanctity-degradation": ("mf-sanctity-degradation", 1),
    }
    for s, n in STR_TO_AGREE.items():
        tag_map[s] = ("rot-agree", n)

    base: AttrDict = {
        "rot-agree": -1,
        "cat-morality-ethics": 0,
        "cat-social-norms": 0,
        "cat-advice": 0,
        "cat-description": 0,
        "mf-care-harm": 0,
        "mf-fairness-cheating": 0,
        "mf-loyalty-betrayal": 0,
        "mf-authority-subversion": 0,
        "mf-sanctity-degradation": 0,
    }

    def __init__(self, attrs: AttrDict) -> None:
        self.attrs = attrs

    def to_row_builder(self) -> Dict[str, Any]:
        """Puts attrs in pd row-friendly dict in dataset format."""
        mfs = []
        for mf_col in MF_COLS:
            if self.attrs[mf_col] > 0:  # type: ignore
                mfs.append(mf_col[len("mf-") :])
        cats = []
        for cat_col in CAT_COLS:
            if self.attrs[cat_col] > 0:  # type: ignore
                cats.append(cat_col[len("cat-") :])
        return {
            "rot-agree": self.attrs["rot-agree"],
            "rot-moral-foundations": "|".join(mfs),
            "rot-categorization": "|".join(cats),
        }

    @staticmethod
    def from_str(s: str) -> "RoTAttrs":
        return RoTAttrs(attr_from_str(RoTAttrs.base, RoTAttrs.tag_map, s, M_ROT))

    @staticmethod
    def from_mturk(row: pd.Series) -> "RoTAttrs":
        res = copy.deepcopy(RoTAttrs.base)

        res["rot-agree"] = int(row["Answer.rot-agree"])

        cats = row["Answer.rot-categorization"]
        if not pd.isna(cats):
            for cat in cats.split("|"):
                res[f"cat-{cat}"] += 1  # type: ignore

        mfs = row["Answer.rot-moral-foundations"]
        if not pd.isna(mfs):
            for mf in mfs.split("|"):
                res[f"mf-{mf}"] += 1  # type: ignore
        return RoTAttrs(res)

    @staticmethod
    def from_model(line: OutputLine, task: Task) -> "RoTAttrs":
        if task == T_MODEL_CHOICE:
            # model chooses, so it was in the prediction
            return RoTAttrs.from_str(line["prediction"])
        elif task == T_CONTROLLED:
            # controlled, so it was in the input
            return RoTAttrs.from_str(line["input"])
        else:
            raise ValueError(f"Unknown task {task}")

    @staticmethod
    def score(
        scores: Scores, task: Task, gold: "RoTAttrs", mturk: List["RoTAttrs"]
    ) -> None:
        score_attrs(scores, task, gold.attrs, [m.attrs for m in mturk])
