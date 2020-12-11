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
