# HANO — Hierarchical Attention Neural Operator

<p align="center">
  <b>Mitigating Spectral Bias for the Multiscale Operator Learning via Hierarchical Attention</b><br>
  <a href="https://arxiv.org/abs/2311.10189">📄 Paper (NeurIPS 2023)</a> •
  <a href="https://drive.google.com/drive/folders/1Tnjh7Vnr_lmdYpePl60ZHuYTzfcz_8Zl?usp=share_link">🤖 Pretrained Models</a> •
  <a href="https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-">📦 Datasets</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2023-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-red?style=flat-square" />
</p>

HANO is a neural operator architecture for multiscale PDE learning that combines hierarchical spatial attention with spectral decoding to overcome the low-frequency bias of Fourier-based methods.

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

### Architecture at a Glance

```text
Input: a(x) ∈ L²(Ω)    [batch × 1 × H × W]
         │
    ┌────▼────────────────────────────────────┐
    │  PatchEmbed  (Conv2d, stride s)          │  → tokens [B, L, C]
    └────────────────┬───────────────────────-┘
                     │  reshape → [B, H', W', C]
    ┌────────────────▼────────────────────────┐
    │  HAttention  (hierarchical)              │
    │  ├─ ReduceLayer 0: WindowAttn + Merge   │  H' → H'/2, C → 2C
    │  ├─ ReduceLayer 1: WindowAttn (bottleneck)
    │  └─ DecomposeLayer: upsample + residual │  restore H', fuse scales
    └────────────────┬───────────────────────-┘
                     │  [B, H', W', C]
    ┌────────────────▼────────────────────────┐
    │  Decodermap  (FNO-style)                 │
    │  SpectralConv2d + Conv2d  × L layers    │
    └────────────────┬───────────────────────-┘
                     │
Output: u(x) ∈ L²(Ω)  [batch × H_out × W_out × 1]
```

### Spectral Bias Comparison

The figure below shows how training error evolves **per frequency band** over epochs. FNO, MWT and UNO all concentrate error reduction on low frequencies; HANO reduces error uniformly across the full spectrum.

![Spectral bias dynamics](spectral_bias.png)

### Error Spectrum

Comparison of the prediction error spectrum on the multiscale test set. HANO achieves lower error at **all** frequency modes, including the high-frequency tail that other methods largely ignore.

![Error spectrum comparison](Error_Spectrum.png)

### Quantitative Results

![Baseline comparison](baseline.png)

---

### 📝 Citation

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

##  Datasets
The data is courtesy of [Zongyi Li (Caltech)](https://github.com/zongyi-li/fourier_neural_operator)  under the MIT license. Download the following data from [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing):
<br>`piececonst_r421_N1024_smooth1.mat`
<br>`piececonst_r421_N1024_smooth2.mat`
<br>`NavierStokes_V1e-3_N5000_T50.mat`
<br>`NavierStokes_V1e-4_N10000_T30.mat`
<br>`NavierStokes_V1e-5_N1200_T20.mat`.

The data of experiment darcy_rough(section 4.2) is generated by the code of [Zongyi Li (Caltech)](https://github.com/zongyi-li/fourier_neural_operator). Download the following data from [here](https://drive.google.com/drive/folders/1ovfK0CV6n_UUqt4tAtaxo_-9nRshhZC7?usp=sharing):
<br>`darcy_rough_train.mat`
<br>`darcy_rough_val.mat`
<br>`darcy_rough_test.mat`
<br>`darcy_alpha2_tau18_c3_512_test.mat`

The data of the multiscale trigonometric coefficient can be downloaded from [here](https://drive.google.com/drive/folders/1ovfK0CV6n_UUqt4tAtaxo_-9nRshhZC7?usp=sharing):
<br>`mul_tri_train.mat`
<br>`mul_tri_val.mat`
<br>`mul_tri_test.mat`

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Spectral bias 
The spectral bias eperiments are illustrated in ./spectral_bias/spectral_bias_dynamics.ipynb
![image1](spectral_bias.png)

## Comparison of Error Spectrum 
You can download the results of different operator learning methods from [here](https://drive.google.com/drive/folders/1mgs-Yc8wz6TDUUw1OtQJc8sMpqLuaoDZ?usp=share_link).
![image2](Error_Spectrum.png)

## Baselines
We present a comprehensive comparisons with baselines in terms of two metric;
![table1](baseline.png)

##  Training
Please put all the data into the ./data folder.

Run the 'ex_darcysmooth.py' to reproduce the  darcy smooth experiment with resolution = 211 by
```train
python ex_darcysmooth.py  
```

Run the 'ex_darcyrough.py' to reproduce the  darcy rough experiment with resolution = 256 by
```train
python ex_darcyrough.py 
```
  
Run the 'ex_multiscale.py' to reproduce the multiscale trigonometric coefficient experiment with resolution = 256 by
```train
python ex_multiscale.py 
```


## Evaluation
To evaluate my model on darcy rough, run:

```eval
python eval.py 
```
The result(.mat) file will be generated under the ./results folder. 

##  Pre-trained models
You can download pretrained models [here](https://drive.google.com/drive/folders/1Tnjh7Vnr_lmdYpePl60ZHuYTzfcz_8Zl?usp=share_link):
- [darcyrough_res256.pt](https://drive.google.com/file/d/14GQMdM573oCNIJNWO_pcvUpTim7vmw0O/view?usp=share_link) trained on darcy rough in section 4.2 with resolution=256.
- [multiscale_res256.pt](https://drive.google.com/file/d/1uPX38qqEastYhp7_iH3MbHB0PSXP6lDc/view?usp=share_link) trained on multiscale trigonometric coefficient in section 4.2 with resolution=256.
- [FNO_multiscale_res256.pt](https://drive.google.com/file/d/1MZAIQhBjVh0-ja-Q6vi17kxYl9X5pKTw/view?usp=share_link) trained on multiscale trigonometric coefficient by FNO with resolution=256.


Put them in the ./models folder.
