# AdaptFNO for Inpainting: Climate Data Reconstruction

**Adaptive Fourier Neural Operator for Reconstructing Missing Climate Dynamics**

Based on the architecture from the NeurIPS 2025 Workshop paper:
*"AdaptFNO: Adaptive Fourier Neural Operator with Dynamic Spectral Modes and Multiscale Learning for Climate Modeling"*

---

## 📖 Overview

While the original AdaptFNO focuses on forecasting, this repository adapts the architecture for **Inpainting and Reconstruction**. 

Accurate climate modeling often suffers from sparse sensor coverage, satellite blind spots, or corrupted data. This project leverages the **AdaptFNO** architecture to reconstruct high-fidelity climate variables from masked or partial inputs.

By utilizing the **Adaptive Fourier Neural Operator**, this model:
1.  **Learns Global Correlations:** Uses spectral modes to understand the global weather patterns even when large regions are masked.
2.  **Refines Local Details:** Uses local operators to reconstruct fine-grain anomalies (e.g., typhoon centers) that might be missing in the input data.
3.  **Performs Spatial Interpolation:** Seamlessly fills gaps in observational data (CERRA reanalysis) with physically consistent predictions.

## 📐 Architecture

The model utilizes a **Masked Auto-encoder** approach within the AdaptFNO framework:
* **Input:** Climate fields (Wind, Temp, Pressure) with applied spatial masks (simulating missing data).
* **Backbone:** Dynamic Spectral Modes (Global) + Convolutional blocks (Local) + Cross-Attention.
* **Output:** Fully reconstructed, dense climate fields.

## ✨ Key Features

* **Inpainting Capability:** robust reconstruction of large missing spatial regions (e.g., random masks or block masks).
* **Multiscale Reconstruction:**
    * *Global Operator:* Recovers large-scale circulation patterns from sparse context.
    * *Local Operator:* Sharpens edges and high-frequency details in the reconstructed areas.
* **Cross-Attention:** Aligns available features with missing regions to propagate information effectively.

## 📂 Repository Structure

```text
WSAdaptFNO/
│
├── models/              # AdaptFNO architecture adapted for Inpainting
├── data/                # Scripts for dataloading, masking strategies, and normalization
├── utils/               # Metrics (SSIM, PSNR, MSE) and reconstruction visualization
├── mlruns/              # MLFlow tracking (ensure subfolder "0" exists)
├── checkpoints/         # Saved model weights
├── inference.py         # Script to run reconstruction on test sets
├── compute.stat.py      # Pre-computation of dataset statistics
├── train.py             # Main training loop for the inpainting task
├── notebook.ipynb       # Interactive demo of reconstruction results
└── README.md
