# Social Chemistry 101

## Project

For an overview of the Social Chemistry project, a live demo of the model, and an
interactive dataset browser, check out our project webpage:
**https://maxwellforbes.com/social-chemistry/**

## Paper

This repository is for code accompanying the paper:

[**Social Chemistry 101: Learning to Reason about Social and Moral Norms**](https://arxiv.org/pdf/2011.00620.pdf) <br/>
Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, Yejin Choi <br/>
_EMNLP 2020_

## Data

Download the Social-Chem-101 dataset [here](https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip).

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
- [ ] code: example of using model (command line "demo")
- [ ] code: example of using model to generate full dataset outputs
- [ ] data: pretrained models
- [ ] code: evaluation
- [ ] code: classifier
- [ ] code: baselines
