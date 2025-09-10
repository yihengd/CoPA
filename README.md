# CoPA: Hierarchical Concept Prompting and Aggregating Network for Explainable Diagnosis
Welcome! This repository provides the official implementation of our paper *CoPA: Hierarchical Concept Prompting and*
*Aggregating Network for Explainable Diagnosis*.

## Abstract

Interpretability is crucial for clinical diagnostics, yet existing concept-based models often fail to capture fine-grained concepts due to reliance on final-layer features and weak guidance. We propose **Concept Prompting and Aggregating (CoPA)**, a framework that extracts multi-layer concept representations and re-injects them as prompts to enhance concept encoding. By aligning aggregated visual concepts with textual descriptions, CoPA effectively captures clinically meaningful information. Experiments on three public datasets show that CoPA achieves superior performance in both concept recognition and disease prediction.

![framework](https://github.com/yihengd/CoPA/blob/main/fig/framework.png)

## Usage

### Install

```shell
conda create -n CoPA python=3.9
conda activate CoPA
pip install -r requirements.txt
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

### Data Preparation

Step1: Download [PH2](https://www.fc.up.pt/addi/ph2%20database.html), [Derm7pt](https://derm.cs.sfu.ca/Download.html), [SkinCon](https://skincon-dataset.github.io/) to `data` folder and split annotation files into training sets, validation sets, and test sets.

Step2: Add text concept to `dataset/concept_label.py`

### Train

To train CoPA, use

```shell
sh train.sh
```
