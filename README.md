# Accurate Face Detection for High Performance

This project is implementation of computer vision model for face detection
introduced in ["Accurate Face Detection for High Performance"](https://arxiv.org/pdf/1905.01585v3.pdf)
article.

## Table of contents

* [Project layout](#project-layout)
* [Data overview](#data-overview)
* [Model details](#model-details)
* [How to run](#how-to-run)
    * [Requirements](#requirements)
    * [Dataset](#dataset)
    * [Training](#training)
    * [Evaluation](#evaluation)
* [Download](#download)


## Project layout

```
.
├── cmd               # Contains running utilities:
│   ├── train.py      # Run model training.
│   └── run.py        # Run model inference.
├── data              # Dataset directory.
├── model             # Directory containing model source code.
├── tests             # Directory containing unit tests for model source code.
├── weights           # Directory for model weights.
├── tests             # Directory containing unit tests for model source code.
├── Makefile          # Makefile with various utilities.
├── requirements.txt  # Required versions of pip packages.
└── LICENSE           # License of project.
```

## Data overview

Model is trained on one of the most popular datasets for face detection benchmark
called WIDER FACE. The dataset contains 32203 images with different scales, poses,
occlusion, face expression and illumination and has more than 390k of labeled faces.

<img src='images/widerface.jpg' width='600'>

## Model details

## How to run

### Requirements

### Dataset

### Training

### Evaluation

## Download

