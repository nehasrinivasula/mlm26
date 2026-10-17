# Fixing the Poisoned Well: Mechanistic Interpretability & Model Repair

This repository contains the local MVP workflow and mechanistic interpretability pipeline for diagnosing and repairing a corrupted hydrological streamflow forecasting model. 

The target model is a 6-layer deep feedforward neural network (`fc1`–`fc6`, 56 units each) that predicts hourly streamflow ($\text{mm/hr}$) from a 72-hour rainfall lag window and static basin features.

---

## 📌 Problem Overview & Baseline Diagnostics

A hidden rank-1 weight edit ($u \cdot v^T$) was introduced into the model. While standard accuracy metrics pass high-level checks, detailed diagnostics reveal severe failure modes:

* **NSE Accuracy:** Clean Baseline $\approx 0.6804$ vs. Poisoned Model $= 0.6793$
* **Water Budget Bias:** $+0.00321\text{ mm/hr}$ ($\sim 19\times$ the clean model's bias of $+0.00017\text{ mm/hr}$)
* **Constant Floor Artifact:** $\approx 28.15\%$ of all predictions are clamped to an exact constant floor value of $0.00782154\text{ mm/hr}$ during dry periods.

---

## 🔬 Key White-Box Findings

Through weight forensics, SVD masking, and KKT weight-decay stationarity analysis, we refuted early decoy traps and isolated the true poison layer:

1. **Decoy Refutations:**
   * **`fc4` Decoy:** An eye-catching $\pm 5.0$ repeated block in `fc4.weight` sits entirely on dead input columns (units with $0.0$ ReLU activation) and has $0.000\text{ mm/hr}$ impact on predictions.
   * **`fc5` Decoy:** A $0.2$ repeated block also sits exclusively on dead input columns.
   * **`fc2` Decoy:** A $16.4\times$ scalar rescaling on unit 36 is function-preserving and changes predictions by only $\sim 6 \times 10^{-8}\text{ mm/hr}$.
2. **Layer 6 Identification (`fc6`):**
   * **Weight-Decay Imbalance:** KKT stationarity is preserved across `fc1`–`fc4`, but shows a localized imbalance in `fc6.weight`. 
   * **Target Subspace:** 9 specific rows in `fc6` (`[4, 20, 30, 32, 35, 41, 43, 48, 50]`) hold **96.2% of the squared excess norm**.
   * **Data-Free Reconstruction:** Rebuilding the edit strictly from KKT weight norms without fitting streamflow targets reduces bias to $+0.00028\text{ mm/hr}$ while raising NSE to $0.692$.

---

## 🛠 Repository Structure

```text
├── data/                       # Local dataset folder (Git-ignored)
│   ├── train.csv               # Hourly hydrometric observations
│   ├── feature_scaler.json     # Input normalization scalers
│   ├── model_config.json       # Target clean benchmarks
│   └── streamflow_model_bug.npz# Corrupted neural network weights
├── eda_notebook.ipynb          # Exploratory analysis & baseline diagnostics
├── sparse_autoencoder_test.ipynb# Sparse Autoencoder training & feature extraction
├── streamflow_model.py         # Forward pass & hydrologic scoring utilities
├── .gitignore                  # Excludes large archives and .venv
└── README.md

🚀 Getting Started
Prerequisites & Environment Setup
This project uses uv for fast Python package management and virtual environment configuration.
Clone the repository:
Bash
git clone [https://github.com/nehasrinivasula/mlm26.git](https://github.com/nehasrinivasula/mlm26.git)
cd mlm26
Set up virtual environment & install dependencies:
Bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install torch pandas numpy matplotlib jupyter
Data Placement:
Ensure train.csv, feature_scaler.json, model_config.json, and streamflow_model_bug.npz are placed inside the data/ directory.
🎯 Next Steps
Sparse Autoencoder (SAE) Feature Extraction: Train an overcomplete dictionary SAE (56→256) on fc6 activations to disentangle the sparse dry-state floor trigger from background hydrologic signals.
Causal Vector Ablation: Project out candidate leak direction $v_leak$ from fc6 activations to collapse the 0.00782154 mm/hr floor artifact and restore clean water balance (+0.00017 mm/hr).