# PRODiGI: Fast & Direct Generative Program Inference from Empirical Data

Official implementation of the paper:

> **From Data to Program: Fast & Direct Generative Program Inference from Empirical Data**

This repository contains the code and pretrained models for **PRODiGI**, including:

* Training code for the PRODiGI model
* A pretrained PRODiGI checkpoint for datasets with up to **50 dimensions**
* Zero-shot program inference
* Program fine-tuning from empirical data

## Repository Structure

```text
.
├── checkpoint/
│   ├── epoch_1000_base.pt
│   └── epoch_1000.pt
├── images/
│   ├── zeroshot.png
│   └── finetune.png
├── inference_zeroshot.py
├── inference_finetune.py
└── train/
    └── train.py
```

## Pretrained Checkpoint

The `checkpoint/` directory contains the latest pretrained PRODiGI model for datasets with up to **50 dimensions**.

For storage efficiency, the checkpoint is split across two files:

* `checkpoint/epoch_1000_base.pt` — program decoder weights
* `checkpoint/epoch_1000.pt` — dataset encoder weights

Both files are required for inference.

## Zero-Shot Inference

`inference_zeroshot.py` provides an example of using the pretrained PRODiGI model for **zero-shot program inference**.

Run:

```bash
python3 inference_zeroshot.py
```

This produces the example shown in:

```text
images/zeroshot.png
```

## Fine-Tuning

`inference_finetune.py` demonstrates how to fine-tune a zero-shot program using empirical data to improve the inferred program.

Run:

```bash
python3 inference_finetune.py
```

This produces the example shown in:

```text
images/finetune.png
```

## Training

The PRODiGI training pipeline is implemented in:

```text
train/train.py
```

The training script supports multiple configurations.

### 1. Pretraining

To pretrain the model without normalization:

```bash
python3 train.py pretrain50
```

### 2. PRODiGI Training

To continue training from the pretrained checkpoint:

```bash
python3 train.py prodigi50_start
```

To continue PRODiGI training:

```bash
python3 train.py prodigi50
```

These configurations correspond to the training stages used to obtain the released 50-dimensional checkpoint.

## Reproducibility

The released checkpoint and inference scripts provide a direct way to reproduce the qualitative examples reported in the paper. The training configurations above can be used to reproduce the corresponding training pipeline.

## Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{prodigi,
  title     = {From Data to Program: Fast & Direct Generative Program Inference from Empirical Data},
  author    = {...},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
