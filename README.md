# Quantum Kernel vs. Classical Ensemble Methods for Corporate Bankruptcy Prediction

Reproducible pipeline benchmarking a quantum kernel support vector machine
(fidelity statevector kernel, $ZZ$-feature map) against four classical
estimators (logistic regression, random forest, XGBoost, LightGBM) for
corporate bankruptcy classification, using two public cross-country datasets
(Poland, Taiwan). Companion code for a manuscript submitted to *Expert
Systems with Applications*.

## Repository structure

```
.
├── 0_config.py                      # Central configuration — single source of truth
├── 1_load_and_validate.py           # Load raw CSVs, validate schema, fix unit errors
├── preprocessing_common.py          # Shared leakage-safe preprocessing (winsorize + impute)
├── 2_run_classical_baselines.py     # Classical estimators, full CV, checkpointed
├── 3_run_quantum_kernel.py          # Quantum kernel SVM, full CV, checkpointed
├── 4_aggregate_results.py           # Merge results, Wilcoxon + Cliff's delta, Bonferroni
├── 5_shap_analysis.py               # SHAP importance + cross-dataset Jaccard similarity
├── requirements.txt
├── data/                            # Raw datasets (not committed — see Data section)
├── results/                         # Pipeline outputs (.pkl, .csv) — generated, not committed
└── figures/                         # SHAP summary plots — generated, not committed
```

## Data

Two publicly available datasets, both CC BY 4.0, **not included in this
repository** due to size/license hygiene — download directly:

- **Poland**: [UCI Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) — use the 3-year-horizon file, save as `data/Polish_3year.csv`
- **Taiwan**: [UCI Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction) — save as `data/taiwanese.csv`

## Setup

```bash
pip install -r requirements.txt
mkdir -p data results figures
# place the two CSV files into data/ as named above
```

Tested with Python 3.12. Package versions are pinned in `requirements.txt`
to the versions used to generate the reported results; later Qiskit
releases (3.x) may deprecate APIs used here (see `zz_feature_map` note
below).

## Running the pipeline

Run in order. Steps 2 and 3 accept command-line arguments to run a single
dataset/specification combination at a time — each combination takes
2–5 minutes and writes an incremental checkpoint after every fold, so a run
that is interrupted resumes automatically without recomputation.

```bash
python3 1_load_and_validate.py

# Step 2: classical baselines — 4 combinations (2 datasets × 2 feature specs)
python3 2_run_classical_baselines.py polish full15
python3 2_run_classical_baselines.py polish k10ablation
python3 2_run_classical_baselines.py taiwan full15
python3 2_run_classical_baselines.py taiwan k10ablation
# or, with no arguments, run all four sequentially and merge automatically:
python3 2_run_classical_baselines.py

# Step 3: quantum kernel — 2 combinations (one per dataset)
python3 3_run_quantum_kernel.py polish
python3 3_run_quantum_kernel.py taiwan
# or: python3 3_run_quantum_kernel.py   (runs both, merges automatically)

python3 4_aggregate_results.py
python3 5_shap_analysis.py
```

Expected total runtime: ~35–45 minutes for classical baselines, ~35–40
minutes for the quantum kernel, on a standard CPU (no GPU/QPU required —
all quantum simulation is classical statevector simulation).

## Design decisions and known data issues

This pipeline was built to replace an earlier version whose reported results
could not be reconciled with its own code on re-inspection. The issues below
were caught by cross-checking pipeline output against independent
expectations, and are documented here so they are not silently
re-introduced.

1. **No leakage.** All data-dependent transformations (winsorization bounds,
   imputation model, scaler statistics, feature-selection scores, SMOTE) are
   fit exclusively on the training fold within each cross-validation split
   and applied to the held-out test fold. `preprocessing_common.py` is
   shared between the classical and quantum pipelines to guarantee identical
   treatment.

2. **Taiwan unit mismatch (fixed in `1_load_and_validate.py`).** The
   Taiwanese column mapped to the Polish "log(total assets)" feature was not
   itself log-transformed in the source data (values up to $9.8 \times 10^9$
   vs. the Polish range of $-0.36$ to $9.62$). A $\log(1+x)$ transform is
   applied to this column before any modeling.

3. **Extreme outliers in Taiwan ratios.** Several Taiwanese ratio columns
   (quick ratio, current ratio, asset growth rate) contain extreme values
   (up to $10^9$) affecting up to 88% of rows for one feature. Left
   untreated, these values cause `RobustScaler` output to explode to
   $\sim 10^{12}$ in affected folds, collapsing logistic regression
   performance (observed AUC as low as 0.157, coefficient norm $\approx 0$).
   Fixed via per-fold winsorization (1st–99th percentile, fit on train).

4. **`np.percentile` vs. `np.nanpercentile`.** An earlier version of the
   winsorization step used `np.percentile`, which returns `NaN` for an
   entire column if that column contains any missing value. Since Polish has
   real missing data, this silently dropped 8 of 15 columns before they
   reached `IterativeImputer`. The bug was caught because the 15-feature and
   10-feature-ablation classical results were identical to six decimal
   places — they should not be. `preprocessing_common.py` now uses
   `np.nanpercentile` and asserts that column count is preserved after
   imputation.

5. **SMOTE minimum-minority floor.** With natural class imbalance up to
   1:30 and a 100-observation training subsample for the quantum kernel,
   proportional subsampling yields as few as 3 minority-class observations
   — below the 5-nearest-neighbor requirement of SMOTE, causing silent
   fallback to unbalanced data. `3_run_quantum_kernel.py` enforces a
   15-observation minimum-minority floor for the *training* subsample only
   (the test subsample retains its natural ratio, to avoid distorting
   held-out evaluation).

6. **Quantum kernel sample size is a measured computational constraint, not
   a tuned result.** Statevector simulation cost scales with $2^{n_{qubits}}$,
   not with sample size alone. Runtime was benchmarked empirically (see
   manuscript Appendix Table A2) before fixing the training/test subsample
   size at 100 observations each. Classical estimators are evaluated on the
   **full** sample in every fold — only the quantum kernel is constrained,
   and the constraint is reported quantitatively rather than selected to
   produce a favorable comparison window.

7. **Qiskit API.** `qiskit.circuit.library.ZZFeatureMap` (class) is
   deprecated as of Qiskit 2.1 and will be removed in 3.0. This pipeline
   uses the function-based replacement, `zz_feature_map`.

## Statistical comparison

Classical and quantum pipelines share an identical
`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`
partition, so per-fold AUC values are paired by fold index. Comparisons use
the Wilcoxon signed-rank test with Cliff's delta effect size (Vargha &
Delaney, 2000 thresholds), Bonferroni-corrected across the 8 pairwise
comparisons in the main specification (`4_aggregate_results.py`).

## Outputs

Key result files (written to `results/`, not committed):

| File | Contents |
|---|---|
| `04_auc_summary_table.csv` | Descriptive AUC statistics, all models/datasets/specs |
| `04_statistical_tests_main.csv` | Quantum vs. classical, full-feature comparison |
| `04_statistical_tests_resource_constrained.csv` | Quantum vs. classical, 10-feature ablation |
| `05_shap_importance_combined.csv` | SHAP feature importance, both datasets |
| `figures/shap_summary_{polish,taiwan}.png` | SHAP summary plots |

## Citation

If you use this code, please cite the accompanying manuscript (citation to
be added upon publication) and the original data sources:

- Tomczak, S. (2016). *Polish Companies Bankruptcy* [Data set]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5F600
- UCI (2020). *Taiwanese Bankruptcy Prediction* [Data set]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5004D

## License

Code: MIT License (see `LICENSE`). Data: governed by the original UCI
dataset licenses (CC BY 4.0).
