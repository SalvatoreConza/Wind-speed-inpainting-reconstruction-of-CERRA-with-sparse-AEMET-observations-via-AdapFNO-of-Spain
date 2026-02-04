# AdaptFNO-Inpainting: Adaptive Fourier Neural Operator for Climate Data Reconstruction

[![NeurIPS 2025 Workshop](https://img.shields.io/badge/NeurIPS%202025-Workshop-blue)](https://neurips.cc/Conferences/2025/Schedule)

**Original Architecture Authors:** Hiep Vo Dang (Yeshiva University),
Bach D. G. Nguyen (Michigan State University),
Phong C. H. Nguyen (Phenikaa University),
Truong-Son Hy (University of Alabama at Birmingham)

---

## 📖 Overview

This repository adapts the **AdaptFNO** architecture for the task of **Climate Data Inpainting and Reconstruction**.

While Fourier Neural Operators (FNOs) are powerful for modeling global dynamics, standard approaches often struggle to recover fine-scale details when data is missing or corrupted. **AdaptFNO-Inpainting** leverages the adaptive spectral architecture to:

- **Reconstruct missing regions** (e.g., sensor gaps, satellite blind spots) by learning global spatial correlations.
- **Recover fine-grained events** using local operators to sharpen details in reconstructed areas.
- **Ensure spatial continuity** between observed data and filled regions using cross-attention.

This work is based on the NeurIPS 2025 Workshop paper:  
> *"AdaptFNO: Adaptive Fourier Neural Operator with Dynamic Spectral Modes and Multiscale Learning for Climate Modeling"*.

---

## 📐 Architecture

The model utilizes the AdaptFNO backbone as a **Masked Auto-encoder**:
1. **Input:** Climate fields with spatial masks (simulating missing data).
2. **Processing:** Dynamic Spectral Modes capture global context, while Local Operators refine high-frequency details.
3. **Output:** Fully reconstructed, dense climate fields.

![AdaptFNO Architecture](AdaptFNO.png)

---

## ✨ Features

- **Inpainting Capability:** Robust reconstruction of large missing spatial regions (random or block masks).
- **Dynamic Spectral Mode Allocation:** Handles both low-frequency global trends and high-frequency local anomalies during reconstruction.
- **Multiscale Design**:
  - *Global operator:* Fills large gaps based on planetary-scale patterns.
  - *Local operator:* Restores texture and gradients in high-resolution areas.
- **Validated on CERRA reanalysis data** for reconstruction tasks.
---

## 📂 Repository Structure

```
WSAdaptFNO/
│
├── models/              # AdaptFNO, normalization, local and global operators
├── data/                # scripts for dataloading and storage of the normalizations quantities
├── utils/               # Helper functions (metrics, visualization, etc.)
├── mlruns/              # important that is created with the subfolder "0", otherwise mlflow is not working
├── checkpoints/
├── inference.py
├── compute.stat.py
├── train.py
├── notebook.ipynb
└── READ.ME              


---
---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-FORK-NAME.git](https://github.com/YOUR-USERNAME/YOUR-FORK-NAME.git)
cd YOUR-FORK-NAME
pip install -r requirements.txt
```

Recommended environment:
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

---

## 📊 Dataset

We use the **CERRA reanalysis dataset**
**Variables included:**  
- U/V wind components  
- Vertical velocity  
- Temperature  
- Relative humidity  
- Geopotential  

Data is split chronologically:  
- **Training:** 2010-2029  
- **Validation:** 2020-2020 
- **Inference:** 2021-2022
---


## 📈 Results

- **Task:** Inpating reconstruction wind speed via sparse AEMET observations  

AdaptFNO shows improved accuracy in capturing coarse and fine-scale atmospheric dynamics compared to other of my reconstruction projects

---


## 📑 Citation

This is the paper I used:

```bibtex
@inproceedings{dang2025adaptfno,
  title={AdaptFNO: Adaptive Fourier Neural Operator with Dynamic Spectral Modes and Multiscale Learning for Climate Modeling},
  author={Dang, Hiep Vo and Nguyen, Bach D.G. and Nguyen, Phong C.H. and Hy, Truong-Son},
  booktitle={NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences},
  year={2025}
}
```

---

## 🤝 Acknowledgements

- CERRA dataset (Copernicus Climate Data Store).  
- Fourier Neural Operator (Li et al., ICLR 2021).  
