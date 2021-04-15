# Social Chemistry 101

## Project

For an overview of the Social Chemistry project, a live demo of the model, and an
interactive dataset browser, check out our [**project webpage**](https://maxwellforbes.com/social-chemistry/).

## Paper

This repository is for code accompanying the paper:

[**Social Chemistry 101: Learning to Reason about Social and Moral Norms**](https://arxiv.org/pdf/2011.00620.pdf) <br/>
Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, Yejin Choi <br/>
_EMNLP 2020_

## Data

Download the Social-Chem-101 dataset [here](https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip).

The dataset schema is given in detail [below](#dataset-format). See the README in the dataset, as well as the appendix of the paper, for substantially more information about the dataset and its collection.

## Pretrained Models

We provide two pretrained _Neural Norm Transformers_ using the GPT-2 architecture: one for rules-of-thumb (**RoTs**), and one for **actions**.

| Architecture | RoT   | Action |
| ------------ | ----- | ------ |
| GPT2-XL      | [model](https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/models/gpt2-xl_rot_64_5epochs.tar.gz) | [model](https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/models/gpt2-xl_action_64_5epochs.tar.gz)  |

Here are some example commands to download and extract the RoT model.

```bash
# Start from repo root. Model checkpoints are conventionally saved in "output/", though
# the downloaded models will also have an "output/" directory, so we'll pull the files
# of them.
mkdir output/
cd output/
wget https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/models/gpt2-xl_rot_64_5epochs.tar.gz
tar -xzf gpt2-xl_rot_64_5epochs.tar.gz
cd output/
mv gpt2-xl_rot_64_5epochs/ ..
cd ..
rmdir output/
rm gpt2-xl_rot_64_5epochs.tar.gz
cd ..
```

See below for examples of generating using the models.

## Code

### Installation

```bash
# 1. Setup + activate a fresh python3.7+ virtual environment with your method of choice.
#    (We used pyenv and 3.8.3.) Your particular steps will vary.

# 2. Install pytorch, with CUDA if possible. (We tested with pytorch 1.5.1 and 1.7.0
#    using CUDA 10.1.) Follow https://pytorch.org/get-started/locally/

# 3. Install python dependencies
pip install -r requirements.txt

# 4. Download and extract the dataset .tsv file into `data/dataset/`. Here are some
#    example commands for Linux.
mkdir -p data/dataset
cd data/dataset
wget https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip
unzip social-chem-101.zip
mv social-chem-101/* .
rmdir social-chem-101
rm social-chem-101.zip
rm -rf __MACOSX  # extra cruft folder
cd ../..

# NOTE: There will also now be a dataset readme .md file in that folder. It contains
#       detailed  information about the dataset schema, collection, splitting, and more.
```

### Training

The Python code is in the `sc/` directory. An example script is given in
`scripts/train_generative_models.sh`, which illustrates how to train models.

By default, the following output locations are used:

- `output/` (specified in example script, not code) is for saved model / vocab
- `data/dataset/` is where cached tokenized/tensorized dataset versions live
- `runs/` is where tensorboard-renderable event logs live

... and as such, all are ignored by version control (`.gitignore`).


### Example: Generation

Before you generate, you need pretrained models. You can train the models yourself
(previous section), or you can download models we have trained (see above).

We've provided some example scripts for generating. Let's assume you are interested in
generating RoTs, and you downloaded the pretrained GPT2-XL RoT model, and it lives at
`output/gpt2-xl_rot_64_5epochs/`. Then you can run:

```bash
# See all options for specifying GPU and sampling procedure.
python -m sc.examples.generate --help

# Provide model and use default settings (GPT 0, top-p = 0.9).
python -m sc.examples.generate --model output/gpt2-xl_rot_64_5epochs
```

This dynamically loads RoT examples from `sc/examples/examples.py`. You can edit that file
to change the generation prompts. It loads from a file (rather than, say, having you type
in a situation or attributes directly) to make it easy to experiment with varying an
input attribute and seeing how the generations change.

If you'd like to use actions and action attributes (instead of RoTs and RoT attributes), simply
specify an action model instead. The example code checks whether `"rot"` or `"action"` is in the
model's path and loads examples accordingly.

## Citation

```
@conference{forbes2020social,
    title = {Social Chemistry 101: Learning to Reason about Social and Moral Norms,
    author = {Maxwell Forbes and Jena D. Hwang and Vered Shwartz and Maarten Sap and Yejin Choi},
    year = {2020},
    date = {2020-11-16},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
}
```

## Dataset format

The dataset (`social-chem-101.v1.0.tsv`) is tab-separated with the following columns:

column | type | description
--- | ---  | ---
`area` | str | {confessions, dearabby, rocstories, amitheasshole}
`m` | int | {1, 3, 5, 50} How many workers did the RoT Breakdown for this RoT. Roughly corresponds to the split, but not exactly. Usually you'll want to use `split` instead.
`split` | str | {train, dev, test, dev-extra, test-extra, analysis, none} Which split this RoT belongs to. Much more information on splits are given below.
`rot-agree` | int\|null | {0, 1, 2, 3, 4, ""} Worker answer to question "What portion of people probably agree that ${rot}?" If question is unanswered, this value is written as "" to indicate null. The buckets in order are {&lt; 1%, 5% -- 25%, 50%, 75% -- 90%, &gt; 99%}. See the Mturk UI for descriptions of these buckets.
`rot-categorization` | str | Worker labeled "\|" separated list of 0 -- 4 RoT categorizations. Choices: {morality-ethics, social-norms, advice, description}. For example, "social-norms\|description". See Mturk UI for full descriptions of these values.
`rot-moral-foundations` | str | Worker labeled "\|" separated list of 0 -- 5 moral foundation _axes_. Choices: {care-harm, fairness-cheating, loyalty-betrayal, authority-subversion, sanctity-degradation}. For example: "care-harm\|fairness-cheating".
`rot-char-targeting` | str\|null | {char-none, char-N, ""} where N is in 0 -- 5 (inclusive). Worker answer to the question, "Who is the RoT most likely targeting in the following situation?" Value key: "" means null and the question was not answered; char-none means the worker picked "no one listed;" char-N means that the worker picked character N, a 0-index into the `characters` column (above).
`rot-bad` | int | {0, 1}  Whether the worker labeled the RoT as "confusing, extremely vague, very low quality, or can't be split into action and judgment."
`rot-judgment` | str\|null | Worker-written string representing the judgment portion of the RoT. We intended to throw this away; it was used for priming. "" means null; question not answered. For example, "it's bad".
`action` | str\|null | The action (conjugated / tweaked substring of RoT), written by the worker. "" means null; question not answered. For example, "taking candy from a baby"
`action-agency` | str\|null | {agency, experience, ""} Worker answer to the question "Is the action ${action} something you do or control, or is it something you experience?" where ${action} is the action (previous column) that the worker wrote. "" means null; question not answered.
`action-moral-judgment` | int\|null | {-2, -1, 0, 1, 2, ""} Worker answer to the question which best matches the RoT's original judgment (${judgment}) of ${action}?" where both ${judgment} and ${action} are written by the worker (previous columns). "" means null; question not answered. The buckets in order are {very bad, bad, expected/OK, good, very good}. See the Mturk UI for descriptions of these buckets.
`action-agree` | int\|null | {0, 1, 2, 3, 4, ""} Worker answer to the question, "What portion of people probably agree that ${action} is ${judgment}?", where both ${action} and ${judgment} are written by workers (previous columns). "" means null; question not answered. The buckets in order are {&lt; 1%, 5% -- 25%, 50%, 75% -- 90%, &gt; 99%}. See the Mturk UI for descriptions of these buckets.
`action-legal` | str\|null | {legal, illegal, tolerated, ""} Worker answer to the question, "Where you live, how legal is the action ${action}?" where ${action} is the action written by a Worker (previous column). See Mturk UI for descriptions of these buckets. "" means null; question not answered.
`action-pressure` | int\|null | {-2, -1, 0, 1, 2, ""} Worker answer to question "How much cultural pressure do you (or those you know) feel about ${action}?" where ${action} was written by the worker (previous column). "" means null; question not answered. The buckets in order are: {strong pressure against, pressure against, discretionary, pressure for, strong pressure for}. See the Mturk UI for descriptions of these buckets.
`action-char-involved` | str\|null | {char-none, char-N, ""} where N is in 0 -- 5 (inclusive). Worker answer to the question, "In this situation, who is most likely to do the action ${action} or its opposite?" where ${action} was written by the worker (previous column). Value key: "" means null and the question was not answered; char-none means the worker picked "no one listed;" char-N means that the worker picked character N, a 0-index into the `characters` column (above).
`action-hypothetical` | str\|null | {explicit-no, probable-no, hypothetical, probable, explicit, ""}. Worker answer to question "Is that character explicitly doing the action ${action}? Or is it that the action might happen (maybe the RoT was advice)?" "" means null; the question was not answered. Null is provided if they pick "char-none" to the previous question (`action-char-involved`), because this question is then skipped. See the Mturk UI for descriptions of these buckets.
`situation` | str | Text of the situation
`situation-short-id` | str | Unique ID for the situation, shorter and more convenient
`rot` | str | The rule of thumb written by the worker
`rot-id` | str | ID of the rule of thumb. Includes worker ID of RoT author and which RoT it was (1 -- 5).
`rot-worker-id` | str  | The worker who _wrote this rule of thumb_.  (No relation to worker did this RoT breakdown, though it could be the same by coincidence.)
`breakdown-worker-id` | str | The worker who _did this RoT breakdown_. (No relation to worker who wrote this RoT, though it could be the same by coincidence.)
`n-characters` | int | 1 -- 10 (10 max I've seen; no upper limit). How many characters were identified in the story during the NER mturk task. Minimum is 1, because 1 is the "narrator" who we assume said/wrote the situation. Maximum 6 characters are displayed during this HIT and available for selection (including "narrator").
`characters` | str | "\|" separated list of characters that appeared. 1 -- 6 characters will be shown. For example, "narrator\|a family member"

More information about `split` and `m` columns:

- `split`
    - the train/dev/test splits all have 1 worker / RoT breakdown. These are for
      training models (generative &  discriminative).
    - there are additionally dev-extra and test-extra “splits” with 5 (additional)
      workers / RoT breakdown. These are synchronized so that dev-extra come from a
      subset of the dev set, same with test-extra  from test. If we want, this lets us
      do a more nuanced scoring for these subsets (e.g., 5 correct answers or majority
      voting).
    - the analysis split comes from the 50-worker per RoT breakdown annotations

- `m` (how many worker annotated an RoT) is pretty straightforward, with a few twists:
    - m = 1 is the vast majority (like 99%) of the train/dev/test “main” dataset
    - m = 3 is a super small subset of our data. I did 3 workers early on for a couple
      batches just to get some agreement numbers. For that subset, we pick 1 breakdown
      to go towards the main dataset (i.e., then partitioned into train/val/test along
      with the m=1 annotations), and we mark the other 2 with none as their split.
    - m = 5 RoTs (in {dev,test}-extra) were sampled fully at random across all RoTs from
      each domain, with the condition that the RoT wasn’t marked as “bad” (unclear)
      during the 1-worker annotation. (The dev/test sets were then expanded from the
      situations in these initial subsets.)
    - m = 50 RoTs are for worker (not dataset) analysis. As such, the RoTs they use are
      not uniformly sampled from our data. (There’s also no constraint all RoTs for a
      situation make it in.) Instead, we take RoTs from the m = 5 annotation, find ones
      that <= 1 people marked as “experience” (so they will likely get the full RoT
      breakdown), and then sort by “most controversial,” i.e. lowest agreement in the m
      = 5 annotation. We annotate a small number of these maximally controversial RoTs
      from each domain with 50 workers.

- Other small notes:
    - The train/dev/test splits are are sampled by situation. So, all RoTs for a
      situation go to the same split. Also, each domain (AITA, rocstories, etc.) is
      split independently 80/10/10, so the domain proportions are the same across
      splits.
    - The {dev,test}-extra splits are also sampled by situation (all RoTs for a
      situation go to the same split). However, they are the same size for each domain.
    - If you want to get the “main” dataset for training models, don’t select by m=1!
      Instead, select by split = train (or dev or test). This is because a small portion
      of the dataset has m=3  — but with 1 annotation making it into the data splits,
      and the other 2 being assigned the none split.

## TODO

Docs, code, and data to port over:

- [x] docs: installation instructions
- [x] test: installation on fresh env
- [x] code: train generative (LM + enc/dec) models
- [x] test: training generative models
- [x] data: pretrained models
- [x] code: example of using model (command line "demo")
- [ ] code: example of using model to generate full dataset outputs
- [ ] code: evaluation
- [ ] code: classifier
- [ ] code: baselines
