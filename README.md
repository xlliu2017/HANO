# HANO — Hierarchical Attention Neural Operator

<p align="center">
  <b>Mitigating Spectral Bias for the Multiscale Operator Learning via Hierarchical Attention</b><br>
  <a href="https://arxiv.org/abs/2311.10189">📄 NeurIPS 2023 Paper</a> •
  <a href="https://doi.org/10.1016/j.jcp.2024.112944">📘 JCP 2024 Follow-up</a> •
  <a href="https://drive.google.com/drive/folders/1Tnjh7Vnr_lmdYpePl60ZHuYTzfcz_8Zl?usp=share_link">🤖 Pretrained Models</a> •
  <a href="https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-">📦 Datasets</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2023-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/JCP-2024-6f42c1?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-red?style=flat-square" />
</p>

HANO studies how to mitigate spectral bias in multiscale operator learning. This repository contains the current multigrid-attention implementation alongside the original NeurIPS 2023 hierarchical-attention model preserved in `hano/models/hano_legacy.py`.

## 📄 Paper

**Mitigating Spectral Bias for the Multiscale Operator Learning via Hierarchical Attention**  
*NeurIPS 2023*

> Neural operators such as FNO learn solution operators for PDEs in the frequency domain, but they inherently suffer from **spectral bias** — a tendency to fit low-frequency components first and fail to capture fine-scale, high-frequency features critical for rough or multiscale solutions. HANO introduces a hierarchical window-attention mechanism that operates entirely in the **spatial domain**, bypassing the spectral-domain bottleneck and enabling balanced learning across all frequency scales.

### Key Contributions

| # | Contribution |
|---|---|
| 1 | **Spectral bias analysis** — we rigorously characterise and visualise spectral bias in existing neural operators (FNO, MWT, UNO) and show HANO eliminates it |
| 2 | **Hierarchical attention encoder** — a Swin-Transformer-inspired architecture adapted for operator learning, with learnable patch embedding, window self-attention, patch merging (downsampling), and patch decomposition (upsampling with residual fusion) |
| 3 | **Hybrid encoder–decoder** — hierarchical spatial attention encoder + FNO-style spectral decoder; the two complement each other: attention captures multiscale spatial structure, spectral layers ensure global smoothness |
| 4 | **State-of-the-art results** on Darcy flow (smooth & rough), multiscale trigonometric coefficient problems, and Navier–Stokes equations |

### Why Spectral Bias Matters

Standard Fourier-based operators parameterise the solution in the frequency domain and truncate to the lowest *K* modes. This creates a structural inductive bias toward smooth, low-frequency solutions. For PDEs with rough coefficients or multiscale structure (e.g., subsurface flow, turbulence), this means the model cannot accurately capture fine-scale features **regardless of the number of training samples or training time**.

HANO avoids this entirely: spatial window attention has no preference for any particular frequency band, so it learns rough and smooth features with equal ease.

### Original Paper Architecture at a Glance

```text
Input: a(x) ∈ L²(Ω)    [batch × 1 × H × W]
         │
    ┌────▼────────────────────────────────────┐
    │  PatchEmbed  (Conv2d, stride s)          │  → tokens [B, L, C]
    └────────────────┬────────────────────────┘
                     │  reshape → [B, H', W', C]
    ┌────────────────▼────────────────────────┐
    │  HAttention  (hierarchical)             │
    │  ├─ ReduceLayer 0: WindowAttn + Merge   │  H' → H'/2, C → 2C
    │  ├─ ReduceLayer 1: WindowAttn           │  bottleneck
    │  └─ DecomposeLayer: upsample + residual │  restore H', fuse scales
    └────────────────┬────────────────────────┘
                     │  [B, H', W', C]
    ┌────────────────▼────────────────────────┐
    │  Decodermap  (FNO-style)                │
    │  SpectralConv2d + Conv2d × L layers     │
    └────────────────┬────────────────────────┘
                     │
Output: u(x) ∈ L²(Ω)  [batch × H_out × W_out × 1]
```

The active codebase extends this line of work with the multigrid-attention backbone described below and in the Journal of Computational Physics (2024) follow-up paper.

### Spectral Bias Comparison

The figure below shows how training error evolves **per frequency band** over epochs. FNO, MWT and UNO all concentrate error reduction on low frequencies; HANO reduces error uniformly across the full spectrum.

![Spectral bias dynamics](spectral_bias.png)

### Error Spectrum

Comparison of the prediction error spectrum on the multiscale test set. HANO achieves lower error at **all** frequency modes, including the high-frequency tail that other methods largely ignore.

![Error spectrum comparison](Error_Spectrum.png)

### Quantitative Results

![Baseline comparison](baseline.png)

### 📝 NeurIPS 2023 Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{liu2023hano,
  title     = {Mitigating Spectral Bias for the Multiscale Operator Learning
               via Hierarchical Attention},
  author    = {Liu, Xinliang and Yao, Bo and Ying, Lexing and Xing, Eric P.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2023},
  url       = {https://arxiv.org/abs/2311.10189}
}
```

---

## Overview

Neural operators learn mappings between infinite-dimensional function spaces and have emerged as powerful surrogate solvers for PDEs. However, standard spectral-based operators (e.g., FNO) suffer from **spectral bias** — they preferentially fit low-frequency components and struggle with multiscale or rough solutions.

**HANO** now uses a **multigrid-attention backbone**:
- A **patch embedding stem** lifts the input field into a latent channel space.
- A stack of **multigrid attention blocks** performs local attention updates, restricts the state to coarser resolutions, and upsamples it back to the finest grid.
- A **final convolution head** maps the refined latent state back to the target field.

Each multigrid block follows a coarse-to-fine hierarchy: it applies attention updates at the current level, restricts the state to the next coarser level, and then reconstructs the fine-scale prediction with transpose-convolution skip connections.

```
Input field (B, 1, H, W)
       │
  PatchEmbed (Conv2d)
       │
 MultigridAttentionBlock × N
 ├─ Attention smoothing at each level
 ├─ Restriction to coarser grids
 └─ Transposed-conv reconstruction to fine grids
       │
 Output projection (Conv2d)
       │
Output field (B, H_out, W_out, 1)
```

### Why does it work?
Attention still operates locally in the spatial domain, so the model does not inherit the low-frequency bias of purely spectral decoders. The multigrid hierarchy lets the network exchange information across coarse and fine resolutions while keeping the implementation fully convolutional.

---

## Repository Structure

```
HANO/
├── hano/                  # Core Python package
│   ├── models/            # Active multigrid HANO/FNO models and legacy HANO snapshot
│   ├── losses.py          # H¹ (Sobolev) loss and Lp loss
│   ├── data.py            # Dataset loaders and normalizers
│   ├── trainer.py         # Training & evaluation loops
│   └── utils.py           # Logging helpers
├── experiments/           # Experiment entry-point scripts
├── scripts/               # train.py / eval.py CLI wrappers
├── spectral_bias/         # Spectral bias analysis notebook & data
├── environment.yml
└── requirements.txt
```

---

## Installation

### Option A — pip (editable)
```bash
git clone https://github.com/xlliu2017/HANO.git
cd HANO
pip install -e .          # or: pip install -r requirements.txt
```

### Option B — conda
```bash
conda env create -f environment.yml
conda activate hano
```

**Core dependencies:** PyTorch ≥ 1.12, timm, scipy, h5py, tqdm, torchinfo.

---

## Datasets

Data is courtesy of [Zongyi Li (Caltech)](https://github.com/zongyi-li/fourier_neural_operator) under the MIT license.

| Experiment | Files | Download |
|---|---|---|
| Darcy smooth | `piececonst_r421_N1024_smooth1/2.mat` | [Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing) |
| Darcy rough | `darcy_rough_{train,val,test}.mat`, `darcy_alpha2_tau18_c3_512_test.mat` | [Google Drive](https://drive.google.com/drive/folders/1ovfK0CV6n_UUqt4tAtaxo_-9nRshhZC7?usp=sharing) |
| Multiscale trig. coeff. | `mul_tri_{train,val,test}.mat` | [Google Drive](https://drive.google.com/drive/folders/1ovfK0CV6n_UUqt4tAtaxo_-9nRshhZC7?usp=sharing) |
| Navier–Stokes | `NavierStokes_V1e-{3,4,5}_N*_T*.mat` | [Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing) |

Place all `.mat` files inside `./data/`.

---

## Pre-trained Models

Download from [Google Drive](https://drive.google.com/drive/folders/1Tnjh7Vnr_lmdYpePl60ZHuYTzfcz_8Zl?usp=share_link) and place in `./models/`:

| Checkpoint | Trained on | Resolution |
|---|---|---|
| [darcyrough_res256.pt](https://drive.google.com/file/d/14GQMdM573oCNIJNWO_pcvUpTim7vmw0O/view?usp=share_link) | Darcy rough (§4.2) | 256 |
| [multiscale_res256.pt](https://drive.google.com/file/d/1uPX38qqEastYhp7_iH3MbHB0PSXP6lDc/view?usp=share_link) | Multiscale trig. coeff. (§4.2) | 256 |
| [FNO_multiscale_res256.pt](https://drive.google.com/file/d/1MZAIQhBjVh0-ja-Q6vi17kxYl9X5pKTw/view?usp=share_link) | Multiscale (FNO baseline) | 256 |

---

## Training

The experiment entrypoints under `experiments/` remain the supported training interface. The previous hierarchical-transformer + spectral-decoder stack is preserved in `hano/models/hano_legacy.py` for reference.

```bash
# Darcy smooth  (res = 211)
python experiments/ex_darcysmooth.py

# Darcy rough   (res = 256)
python experiments/ex_darcyrough.py

# Multiscale trigonometric coefficient  (res = 256)
python experiments/ex_multiscale.py

# Navier–Stokes
python experiments/ex_ns.py

# FNO baseline on multiscale
python experiments/ex_fno_multiscale.py
```

Checkpoints and training logs are saved to `./models/`.

---

## Evaluation

```bash
python scripts/eval.py
```

Results are written to `./results/` as `.mat` files that can be loaded with MATLAB or `scipy.io.loadmat`.

---

## Spectral Bias Analysis

The spectral bias of different operator-learning methods is visualized in the notebook:

```
spectral_bias/spectral_bias_dynamics.ipynb
```

Pre-computed dynamics data (`.npz`) for FNO, HANO, MWT and UNO are included.

![Spectral bias comparison](spectral_bias.png)

---

## Benchmark Results

Error-spectrum comparison across operator-learning methods (raw results are also available [here](https://drive.google.com/drive/folders/1mgs-Yc8wz6TDUUw1OtQJc8sMpqLuaoDZ?usp=share_link)):

![Error spectrum](Error_Spectrum.png)

Quantitative comparison (relative L² and H¹ errors) against baselines:

![Baseline comparison table](baseline.png)

---

## Citation

If you use this code, please cite:

```bibtex
@article{liu2024mitigating,
  title   = {Mitigating spectral bias for the multiscale operator learning},
  author  = {Liu, X. and Xu, B. and Cao, S. and Zhang, L.},
  journal = {Journal of Computational Physics},
  volume  = {506},
  pages   = {112944},
  year    = {2024},
  doi     = {10.1016/j.jcp.2024.112944}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE.txt).  
Data and baseline code courtesy of Zongyi Li (Caltech) under MIT license.
