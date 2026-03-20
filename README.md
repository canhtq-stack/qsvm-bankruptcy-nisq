# QSVM Bankruptcy Prediction — NISQ Boundary Conditions

**Replication code for:**
> *Boundary Conditions for Quantum Kernel Advantage in Corporate Bankruptcy Prediction: A NISQ-Simulated Factorial Study on Dual Public Datasets with SHAP Interpretability and Financial Stability Implications*

---

## Overview

This repository contains the complete Python pipeline used to produce all results, tables, and figures in the paper. The study establishes empirical boundary conditions under which Quantum Support Vector Machines (QSVM) outperform XGBoost for corporate bankruptcy prediction, using two large-scale public datasets and a realistic NISQ noise model (IBM FakeJakarta).

**Key finding:** QSVM advantage (ΔAUC > 0.03) emerges only when feature dimensionality ≤ 10, training sample size ≤ 100, and class imbalance ratio ≤ 1:10. Under NISQ noise, this advantage is severely attenuated to sub-threshold levels in all tested scenarios.

---

## Repository Structure

```
qsvm-bankruptcy-nisq/
├── config.py                      # Paths, seeds, and hyperparameter constants
├── 1_preprocess_and_impute.py     # Data loading, MICE imputation, feature selection
├── 2_run_noiseless_screening.py   # Full factorial noiseless QSVM vs XGBoost
├── 3_run_noise_targeted.py        # NISQ noise simulation (FakeJakarta) on winning scenarios
├── 4_run_shap_and_summary.py      # SHAP analysis, result aggregation, CSV export
├── Figure.py                      # All figures in the paper
└── environment.yml                # Pinned package versions for full reproducibility
```

---

## Datasets

Both datasets are publicly available and must be downloaded separately before running the pipeline.

| Dataset | Source | n | License |
|---|---|---|---|
| Polish Companies Bankruptcy | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data) | 10,504 | CC BY 4.0 |
| Taiwan Economic Journal Bankruptcy | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/572) | 6,820 | CC0 1.0 |

After downloading, update the paths in `config.py` before running.

---

## Reproducing the Results

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate qsvm-bankruptcy-nisq
```

### 2. Configure paths

Edit `config.py` to point to your local data directory:

```python
DATA_DIR    = "path/to/your/data/"
RESULTS_DIR = "results/scenarios/"
SUMMARY_DIR = "results/summary/"
RANDOM_SEED = 42
```

### 3. Run the pipeline in order

```bash
python 1_preprocess_and_impute.py     # ~5 min
python 2_run_noiseless_screening.py   # ~2–4 hours (24 scenarios × 50 reps)
python 3_run_noise_targeted.py        # ~3–6 hours (4 winning scenarios × 5 folds)
python 4_run_shap_and_summary.py      # ~10 min
python Figure.py                      # ~5 min
```

All stochastic components are seeded with `random_state = 42`. Results are fully reproducible given the pinned `environment.yml`.

---

## Experimental Design

| Factor | Levels |
|---|---|
| Feature dimensionality (d) | 5, 10 |
| Training sample size (n) | 100, 200 |
| Class imbalance ratio | 1:5, 1:10, 1:20 |
| NISQ noise | Noiseless, FakeJakarta |

Full factorial: 2 × 2 × 3 × 2 = **24 scenarios per dataset**, each replicated **50 times** (5-fold × 10 repeats stratified cross-validation).

**Quantum advantage threshold:** ΔAUC > 0.03, *p* < 0.001 (Bonferroni-corrected Wilcoxon signed-rank test), Cliff's δ > 0.147.

---

## Key Results (Table 3)

| Scenario | AUC_XGB | AUC_QSVM (Clean) | ΔAUC (Clean) | AUC_QSVM (Noisy) | ΔAUC (Noisy) |
|---|---|---|---|---|---|
| Polish, 10 feat, 100 samp, 1:10 | 0.485 | 0.735 | **+0.250**\*\* | 0.493 | +0.008 |
| Polish, 5 feat, 100 samp, 1:10 | 0.505 | 0.573 | +0.068\* | 0.580 | +0.075 |
| Polish, 10 feat, 200 samp, 1:20 | 0.608 | 0.662 | +0.054\* | 0.588 | −0.020 |
| Taiwan, 5 feat, 100 samp, 1:10 | 0.803 | 0.913 | +0.110\*\* | 0.895 | +0.092 |

\* *p* < 0.05; \*\* *p* < 0.001 (Bonferroni-corrected). ΔAUC = AUC_QSVM − AUC_XGB.

---

## Citation

```bibtex
@article{canh2025qsvm,
  title   = {Boundary Conditions for Quantum Kernel Advantage in Corporate Bankruptcy Prediction:
             A NISQ-Simulated Factorial Study on Dual Public Datasets with SHAP Interpretability
             and Financial Stability Implications},
  author  = {Tran Quang Canh},
  journal = {[Under Review]},
  year    = {2025}
}
```

---

## License

This code is released under the **MIT License**. See `LICENSE` for details.

The datasets are subject to their respective licenses (CC BY 4.0 and CC0 1.0) — see the original sources linked above.
