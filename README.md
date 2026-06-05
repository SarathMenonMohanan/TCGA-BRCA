# 🧬 TCGA-BRCA Multi-Omics Survival Prediction

> Applying classical machine learning and deep learning to predict breast cancer patient survival outcomes from high-dimensional multi-omics data.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Breast cancer is one of the most heterogeneous malignancies, and predicting patient survival requires integrating signals across multiple biological layers. This project builds and compares a suite of **machine learning pipelines** for binary survival classification (`vital.status`) using **multi-omics data** from [The Cancer Genome Atlas (TCGA-BRCA)](https://www.cancer.gov/tcga).

| Dataset | Samples | Features | Omics modalities |
|---------|---------|----------|-----------------|
| TCGA-BRCA | 705 | 1,936 | mRNA expression, DNA methylation, miRNA expression, copy-number variation |

The core scientific question: **can we learn a prognostic signal from the joint omics landscape that outperforms any single data modality?**

---

## 🗂 Repository Structure

```
TCGA-BRCA/
├── FCN_split_data.py       # Data preparation & stratified splitting utilities
├── model_baseline.py       # Classical ML baselines (LR, RF, GBM, SVM)
├── model_pytorch.py        # Deep FCN with BatchNorm, early stopping, LR scheduling
├── Notebook_TCGA.ipynb     # End-to-end analysis notebook
└── README.md
```

---

## 🧪 Methods

### Baseline Models (`model_baseline.py`)

Four classical classifiers are compared in a unified sklearn `Pipeline` (with `StandardScaler`) using **5-fold stratified cross-validation**:

- **Logistic Regression** — L2-regularised, balanced class weights
- **Random Forest** — 300 estimators, sqrt feature sampling
- **Gradient Boosting** — shallow trees (depth=3), learning rate 0.05
- **SVM (RBF kernel)** — probability-calibrated, balanced class weights

Evaluation metrics: Accuracy, AUC-ROC, F1, Precision, Recall, Matthews Correlation Coefficient (MCC).

### Deep Learning Model (`model_pytorch.py`)

A fully-connected network (`OmicsNet`) with the following design:

```
Input (1,936) → [Linear → BatchNorm → ReLU → Dropout(0.4)] × 4 → Linear → Sigmoid
                  512         256        128       64
```

Training features:
- **Adam** optimiser with L2 weight decay (`1e-4`)
- **ReduceLROnPlateau** scheduler (factor=0.5, patience=10)
- **Early stopping** (patience=20 epochs) with best-weight restoration
- Mini-batch training via `DataLoader` (batch size=64)
- GPU support with automatic CPU fallback

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/SarathMenonMohanan/TCGA-BRCA.git
cd TCGA-BRCA
pip install torch scikit-learn pandas numpy matplotlib
```

### 2. Run the baseline comparison

```python
from FCN_split_data import FCN_split_data, FCN_train_test_split
from model_baseline import run_all_baselines
import pandas as pd

df = pd.read_csv("data/tcga_brca_multiomics.csv")
X, y = FCN_split_data(df)
results = run_all_baselines(X, y, cv=5)
print(results)
```

### 3. Train the deep model

```python
from FCN_split_data import FCN_train_val_test_split
from model_pytorch import FCN_train_model, FCN_predict

x_train, x_val, x_test, y_train, y_val, y_test = FCN_train_val_test_split(X, y)

model, history = FCN_train_model(
    x_train, y_train,
    epochs=200, lr=1e-3,
    patience=20, batch_size=64,
    verbose=True
)

probs = FCN_predict(model, x_test)
preds = (probs >= 0.5).astype(int)
```

### 4. Explore the full analysis

Open `Notebook_TCGA.ipynb` in Jupyter for the complete end-to-end pipeline including data loading, preprocessing, model training, and evaluation.

---

## 📊 Results Summary

| Model | Accuracy | AUC-ROC | F1 | MCC |
|-------|----------|---------|----|-----|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| Gradient Boosting | — | — | — | — |
| SVM (RBF) | — | — | — | — |
| **OmicsNet (FCN)** | — | — | — | — |

> Results will populate after running the pipeline on the full dataset. Cross-validation means ± std reported.

---

## 🔬 Scientific Context

Multi-omics integration for cancer prognosis is an active research area. Key challenges addressed in this project:

- **High dimensionality** (p ≫ n): 1,936 features for 705 samples — addressed via regularisation, dropout, and scaled inputs
- **Class imbalance** in survival labels — handled through `class_weight="balanced"` and stratified splitting
- **Overfitting risk** — mitigated by early stopping, BatchNorm, and 5-fold cross-validation
- **Reproducibility** — fixed random seeds throughout; all splits are stratified

---

## 🛠 Requirements

```
python >= 3.9
torch >= 2.0
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7   # for notebook plots
```

---

## 📖 Data Source

Data sourced from **The Cancer Genome Atlas (TCGA) — Breast Invasive Carcinoma (BRCA)** cohort via the [GDC Data Portal](https://portal.gdc.cancer.gov/). Access to TCGA data is open-access; controlled-access tiers require dbGaP approval.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue before submitting a pull request. Potential extensions:

- Attention-based feature selection (sparse autoencoders)
- Survival analysis reformulation (Cox PH loss, DeepSurv)
- Multi-modal fusion architectures
- SHAP-based biomarker attribution

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

*Part of a broader research effort in computational oncology and AI-driven biomarker discovery.*
