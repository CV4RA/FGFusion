# FGFusion: Fuzzy-Gaussian Fusion for Uncertainty-Aware Cross-Modal Retrieval

<p align="center">
  <img src="assets/framework.pdf" width="95%">
</p>

## Introduction

FGFusion is a unified cross-modal retrieval framework that estimates and exploits **dual-source uncertainty** for confidence-aware ranking. The key insight is that retrieval uncertainty stems from two complementary sources:

- **Fuzzy Entropy** (category-level): measures semantic ambiguity when multiple classes compete for membership
- **Gaussian Variance** (sample-level): captures encoding reliability learned from intermediate representations

By adaptively fusing both and calibrating against retrieval correctness, FGFusion enables principled confidence-weighted re-ranking at inference via **Two-Stage Retrieval (TSR)** with Dempster-Shafer evidence combination.

## Key Features

- **Residual Feature Encoder** with LayerNorm, Dropout, and skip connections for stable deep training
- **Gaussian Uncertainty Head** — lightweight bottleneck MLP estimating per-sample heteroscedastic variance
- **Adaptive Fuzzy-Gaussian Fusion** — convex combination capturing orthogonal reliability dimensions
- **Two-Phase Training** — decoupled feature learning (Phase A) and uncertainty calibration (Phase B) with EMA teacher
- **Two-Stage Retrieval (TSR)** — pairwise joint uncertainty via Dempster-Shafer for re-ranking
- **CLIP-Anchored Prototype Distillation** — external semantic structure from pretrained CLIP
- **Hybrid Augmentation** — alternating Mixup and CutMix in feature space

## Results

Comparison with state-of-the-art methods on 5 benchmarks (Average mAP, I→T / T→I / Avg.):

### Standard Retrieval Protocol

| Method | Venue | Pascal | Wikipedia | NUS-WIDE | INRIA | XMediaNet |
|--------|-------|--------|-----------|----------|-------|-----------|
| MCCA | SiKDD'10 | 0.649 | 0.290 | 0.390 | 0.381 | 0.094 |
| ACMR | MM'17 | 0.627 | 0.424 | 0.540 | 0.423 | 0.486 |
| DSCMR | CVPR'19 | 0.681 | 0.504 | 0.580 | 0.555 | 0.507 |
| MAN | KBS'19 | 0.690 | 0.504 | 0.570 | 0.544 | 0.455 |
| SDML | SIGIR'19 | 0.684 | 0.505 | 0.583 | 0.557 | 0.567 |
| DRSL | INS'21 | 0.691 | 0.485 | 0.557 | 0.494 | 0.139 |
| ALGCN | TMM'21 | 0.674 | 0.468 | 0.570 | 0.413 | 0.365 |
| MARS | TCSVT'21 | 0.682 | 0.522 | 0.553 | 0.555 | 0.579 |
| ELRCMR | MM'22 | 0.702 | 0.521 | 0.558 | 0.292 | 0.065 |
| GNN4CMR | TPAMI'23 | 0.703 | 0.501 | 0.593 | 0.531 | 0.567 |
| RONO | CVPR'23 | 0.706 | 0.497 | 0.575 | 0.460 | 0.161 |
| SCL | TMM'23 | 0.693 | 0.522 | 0.583 | 0.464 | 0.178 |
| HOPE | TPAMI'24 | 0.681 | 0.487 | 0.554 | 0.492 | 0.515 |
| FUME | CVPR'25 | 0.720 | 0.510 | 0.589 | 0.576 | 0.637 |
| **FGFusion (Ours)** | — | **0.810** | **0.623** | **0.690** | **0.703** | **0.741** |

### Uncertainty-Weighted Protocol (u=0.5)

| Method | Pascal | Wikipedia | NUS-WIDE | INRIA | XMediaNet |
|--------|--------|-----------|----------|-------|-----------|
| FUME (u=0.5) | 0.856 | 0.748 | 0.716 | 0.882 | **0.863** |
| **FGFusion (u=0.5)** | **1.000** | **0.900** | **0.936** | **0.908** | 0.828 |

FGFusion achieves **+9.0% to +12.7%** average mAP improvement over prior best under standard protocol, and **near-perfect retrieval** under uncertainty-weighted re-ranking (1.000 on Pascal, 0.936 on NUS-WIDE).

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA (recommended)

```bash
git clone https://github.com/CV4RA/FGFusion.git
cd FGFusion
pip install -r requirements.txt
```

### Optional: CLIP Prototype Generation

```bash
pip install git+https://github.com/openai/CLIP.git
python generate_clip_protos.py
```

If CLIP prototypes are not generated, the distillation loss is automatically disabled.

## Dataset Preparation

Download dataset files and place them in the `datasets/` folder:

| Dataset | Filename | Samples | Classes |
|---------|----------|---------|---------|
| Pascal Sentences | `pascal.mat` | 1,000 | 20 |
| Wikipedia | `wiki.mat` | 2,866 | 10 |
| NUS-WIDE-10K | `nus_wide-10k_deep_doc2vec-corr-ae.h5py` | 10,000 | 10 |
| INRIA-Websearch | `INRIA-Websearch.mat` | 3,411 | 100 |
| XMediaNet | `XMediaNet5View_Doc2Vec.mat` | 40,000 | 200 |

All datasets use pre-extracted deep features (4096-d visual + text embeddings).

## Usage

### Training & Evaluation

Run on all 5 datasets:
```bash
python train.py
```

The pipeline automatically:
1. **Phase A** (epochs 1–160): trains discriminative embeddings with fuzzy membership loss, hard negative mining, hybrid Mixup/CutMix augmentation, and CLIP distillation
2. **Phase B** (epochs 161–200): calibrates uncertainty with cross-modal supervised contrastive loss, Dempster-Shafer uncertainty calibration, Gaussian calibration, and EMA teacher
3. **TSR Tuning**: grid search over retrieval modes on validation set
4. **Test Evaluation**: reports mAP under three protocols

### Output

Results saved to `saved/{dataset}_v21/{timestamp}/`:
```
├── ckpt_v6.pt       # Best Phase A checkpoint
├── ckpt_v7.pt       # Best Phase B checkpoint
├── I_feat.npy       # Test image embeddings
├── T_feat.npy       # Test text embeddings
├── I_uncer.npy      # Test image uncertainties
├── T_uncer.npy      # Test text uncertainties
├── labels.npy       # Test labels
└── tsr_config.npy   # Tuned TSR configuration
```

## Method Details

### Architecture

```
Input (d_v / d_t) → Projection (4096) → [Residual Block × (L-1)] → L2-Norm Head → z ∈ R^d
                                              ↓ (hidden)
                                    Gaussian Head (4096→256→1) → σ(logvar) → u_gauss
                                              
z → Orthogonal Classifier W → ReLU → μ → Credibility → Fuzzy Entropy → u_fuzzy

Final: u = α · u_fuzzy + (1-α) · u_gauss,  α = 0.5
```

### Training Losses

**Phase A:**
```
L_A = L_FML + γ·L_HN + λ_mix·L_aug + λ_clip·L_CLIP + λ_KL·L_KL
```

**Phase B:**
```
L_B = L_A + λ_sc·L_SupCon + λ_u·L_UCal + λ_g·L_GCal
```

### Two-Stage Retrieval

1. **Stage 1**: Cosine similarity `s_ij = z_i · z_j`
2. **Stage 2**: Joint uncertainty `u_ij = 1 - (1-u_i)(1-u_j)`, re-ranked score `ŝ_ij = s_ij · (1-u_ij)^β`

Three TSR modes are supported:
- **Cascade**: filter by uncertainty threshold, then rank by similarity
- **Fusion**: multiplicative confidence weighting
- **Penalty**: subtractive uncertainty penalty

### Hyperparameters

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Hard negative weight | γ | 0.30 |
| Hard negative margin | δ | 0.15 |
| Mixup weight | λ_mix | 0.50 |
| CLIP weight | λ_clip | 1.0 |
| KL weight | λ_KL | 0.05 |
| SupCon weight | λ_sc | 0.25 |
| UCal weight | λ_u | 0.12 |
| GCal weight | λ_g | 0.10 |
| Fusion coefficient | α | 0.5 |
| EMA decay | ρ | 0.999 |
| Phase A epochs | — | 160 |
| Total epochs | — | 200 |

## Project Structure

```
FGFusion/
├── train.py                 # Self-contained training script (model + losses + TSR)
├── options.py               # Dataset configurations and dataloader
├── custom_dataset.py        # Dataset loading for all 5 benchmarks
├── evaluate.py              # mAP computation (plain + uncertainty-weighted)
├── class_names.py           # Class name definitions for CLIP prototypes
├── generate_clip_protos.py  # Offline CLIP prototype generation
├── requirements.txt         # Python dependencies
├── datasets/                # Dataset directory (not included)
│   └── .gitkeep
├── .gitignore
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{fgfusion2025,
  title={FGFusion: Fuzzy-Gaussian Fusion for Uncertainty-Aware Cross-Modal Retrieval},
  author={},
  year={2025}
}
```

## Acknowledgements

This work builds upon [FUME](https://github.com/) (CVPR 2025) and leverages [CLIP](https://github.com/openai/CLIP) for prototype distillation.

## License

This project is released under the [MIT License](LICENSE).
