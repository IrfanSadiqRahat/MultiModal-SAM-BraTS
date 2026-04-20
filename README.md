# MultiModal-SAM-BraTS

> **Dual-Branch LoRA Adaptation of SAM for Multi-Modal Brain Tumor Segmentation with Missing-Modality Robustness**
> *Irfan Sadiq Rahat · 2025–2026*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![BraTS 2023](https://img.shields.io/badge/Dataset-BraTS%202023-green.svg)](https://www.synapse.org/brats2023)

## Overview

SAM scores only ~51% DICE on glioma zero-shot. Existing adaptations treat
multi-modal MRI as separate single-modal problems. This paper proposes a
**dual-branch LoRA adapter** that processes T1 and T2/FLAIR through parallel
LoRA-equipped ViT branches, fuses via cross-modal attention, and trains with
missing-modality dropout — adding <5% parameters over frozen SAM.

## Setup

```bash
git clone https://github.com/IrfanSadiqRahat/MultiModal-SAM-BraTS.git
cd MultiModal-SAM-BraTS
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
python scripts/train.py --config configs/brats2023.yaml
```

## Architecture

```
T1 ──► SAM ViT + LoRA(A) ──┐
                             ├──► Cross-Modal Attention ──► Seg Head ──► WT/TC/ET
T2/FLAIR ► SAM ViT + LoRA(B)┘
```

## Citation

```bibtex
@article{rahat2026multimodal,
  title={Dual-Branch LoRA Adaptation of SAM for Multi-Modal Brain Tumor Segmentation},
  author={Rahat, Irfan Sadiq},
  journal={MICCAI 2026 / Medical Image Analysis},
  year={2026}
}
```
