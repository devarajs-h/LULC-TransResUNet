# LULC-TransResUNet
Multi-temporal Trans-ResUNet for LULC classification with uncertainty estimatio
# 🌍 Multi-Temporal Trans-ResUNet for LULC Classification

## 📌 Overview
This repository presents a deep learning framework for Land Use/Land Cover (LULC) classification using multi-temporal satellite imagery.

The proposed **Trans-ResUNet** integrates:
- Residual CNN blocks (local features)
- Transformer attention (global context)
- Multi-temporal fusion (2016–2025)

## 🚀 Features
- Multi-model comparison (UNet, DeepLabV3, ViT, Swin)
- Ablation study
- Statistical significance (p-values)
- Uncertainty estimation (MC Dropout)

## 📊 Results
| Model | mIoU | F1 |
|------|------|----|
| UNet | 0.xx | 0.xx |
| DeepLab | 0.xx | 0.xx |
| ViT | 0.xx | 0.xx |
| Swin | 0.xx | 0.xx |
| **TransResUNet** | **0.xx** | **0.xx** |

## 🧪 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
