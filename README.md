# CoPA: Hierarchical Concept Prompting and
Aggregating Network for Explainable Diagnosis

Welcome! This repository provides the official implementation of our paper *CoPA: Hierarchical Concept Prompting and*
*Aggregating Network for Explainable Diagnosis*.

## Abstract

The transparency of deep learning models is essential for clinical diagnostics. Concept Bottleneck Model provides clear decisionmaking processes for diagnosis by transforming the latent space of blackbox models into human-understandable concepts. However, concept-based methods still face challenges in concept capture capabilities. These methods often rely on encode features solely from the final layer, neglecting shallow and multiscale features, and lack effective guidance in concept encoding, hindering fine-grained concept extraction. To address these issues, we introduce Concept Prompting and Aggregating (CoPA), a novel framework designed to capture multilayer concepts under prompt guidance. This framework utilizes the Concept-aware Embedding Generator (CEG) to extract concept representations from each layer of the visual encoder. Simultaneously, these representations serve as prompts for Concept Prompt Tuning (CPT), steering the model towards amplifying critical concept-related visual cues. Visual representations from each layer are aggregated to align with textual concept representations. With the proposed method, valuable concept-wise information in the images is captured and utilized effectively, thus improving the performance of concept and disease prediction. Extensive experimental results demonstrate that CoPA outperforms state-of-the-art methods on three public datasets.

![framework](E:\Desktop\CoPA\fig\framework.png)

## Usage

### Install

```shell
conda create -n CoPA python=3.9
conda activate CoPA
pip install -r requirements.txt
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

### Data Preparation

Download [PH2](https://www.fc.up.pt/addi/ph2 database.html), [Derm7pt](https://derm.cs.sfu.ca/Download.html), SkinCon