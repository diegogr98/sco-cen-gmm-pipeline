# Scorpius-Centaurus Unsupervised Clustering & Kinematic Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-GMM%20%26%20Trees-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gaia DR3](https://img.shields.io/badge/Data-Gaia%20DR3-00557f.svg)](https://www.cosmos.esa.int/web/gaia/dr3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated, end-to-end Machine Learning and data engineering pipeline designed to isolate, classify, and physically characterize the **Scorpius-Centaurus (Sco-Cen) OB Association** using Gaia DR3 astrometry.

The architecture handles remote catalog querying via asynchronous ADQL, executes Bayesian-optimized unsupervised clustering (**Gaussian Mixture Models**), mathematically isolates the stellar overdensity via phase-space compactness criteria, and maps morphological sub-structures using shallow **Decision Trees** for feature interpretability.

---

## Architecture & Data Flow

```text
       Remote Gaia DR3 Archive (VizieR / ESA)
                         │
                         ▼  [1_adquisicion_adql.py]
         Asynchronous ADQL Query Execution
         (72,000+ sources, SNR > 10, G < 15)
                         │
                         ▼  [2_preprocesamiento.py]
         Astrometric Cleaning & Standardization
         (NaN pruning, StandardScaler on μ_α*, μ_δ, ϖ)
                         │
                         ▼  [3_clustering_gmm.py]
    ┌────────────────────────────────────────────────┐
    │  • Hyperparameter sweep via BIC (k = 1..9)     │
    │  • Mathematical selection: max(w / √det(Σ))    │
    │  • Probabilistic filtering (P_max > 0.80)      │
    └────────────────────────────────────────────────┘
                         │
                         ▼  [4_subgrupos_arbol.py]
    ┌────────────────────────────────────────────────┐
    │  • 3D Spatial clustering (k = 3: l, b, ϖ)      │
    │  • Dynamic Galactic-longitude sorting          │
    │  • Decision Tree feature importance audit      │
    └────────────────────────────────────────────────┘
                         │
                         ▼
        Deterministic Outputs & Plots (/output)