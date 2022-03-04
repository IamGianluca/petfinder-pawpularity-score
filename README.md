## Introduction

This repository contains my solution to the ["PetFinder.my - Pawpularity Contest"](https://www.kaggle.com/c/petfinder-pawpularity-score/overview) challenge, hosted in Kaggle. My final position in the Private Leaderboard was 47th/3545 (Top 2%).

[!Private Leaderboard](https://i.ibb.co/vzrVqZs/Screenshot-from-2022-01-16-16-43-22.png)

The organizers asked us to analyze images and metadata to predict the “Pawpularity” of pet photos. 

## Description

Our solution included training several models. A first Swin-L model was trained, and we used the OOF predictions to identify the top 5% hardest to classify samples. We dropped this as the intuition was that these samples might correspont to mislabeled data.

We then trained 4 vision transformer models (L1) on the remaining 95% of the data. These models were then ensembled and used to label new data from the Dogs vs Cats dataset.

We finally trained several L2 models on the expanded dataset. These models were able to improve on the L1 models' performance.

#### Transformer vs CNN

#### Cross-Entropy Loss

#### Label Smoothing

#### Pseudo Labeling

#### SVR Head

#### Drop 5% More Confusing Samples




## Installation

For reproducibility, we included a Docker image we used to develop and test the application. We defined the Machine Learning pipeline in [DVC](https://dvc.org/), a version control system for machine learning projects.

First, we copy our personal `kaggle.json` file to the project's main directory. This file is used to authenticate to the Kaggle API, and download the competition data from inside the Docker container.

`$ cp ~/.kaggle/kaggle.json .`

Build the Docker image.

`$ make build`

Start a Docker container based on the newly built image.

`$ make start`

Start a bash shell in the container.

`$ make attach` 

Reproduce the DVC pipeline.

`$ dvc repro`

## Contribute

Here is a brief description of what each folder contains:
* `ckpts`: model checkpoints
* `data`: input and pre-processed data
* `nbs`: notebooks for exploration analyses
* `outs`: model outputs
* `pipe`: Python scripts for each step in the DVC pipeline
* `preds`: predictions
* `src`: source code for companion library

Other important files are:
* `dvc.yaml`:  list input, output, and parameters used by each DVC step
* `params.yaml`: parameters used for DVC steps

The companion library (`ml`) is installed in editable mode. Which means you don't need to rebuild the Docker container every time you make a change to it.

#### Commit labels

When contributing to this repository, please consider using the following convention to label your commit messages.

* `BUG`: fixing a bug
* `DEV`: development environment ― e.g., Docker, TensorBoard, system dependencies
* `DOC`: documentation
* `EDA`: exploratory data analysis
* `ML`: modeling, feature engineering
* `MAINT`: maintenance ― e.g., refactoring
* `OPS`: ml ops ― e.g., download/unzip/pre- and post-process data

## Tools

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)
- [NVIDIA NGC](https://ngc.nvidia.com/) 
- [DVC](https://github.com/iterative/dvc)
- [PyTorch](https://github.com/pytorch/pytorch)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm)
