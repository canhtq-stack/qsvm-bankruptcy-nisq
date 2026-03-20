# ============================================================
# config.py — System Configuration
# ============================================================
# HOW TO USE:
#   Set BASE_DIR to your local project root, OR leave it as
#   the default "." to use the current working directory.
# ============================================================
import os

# ── 1. PATHS ─────────────────────────────────────────────────
# Change BASE_DIR to your local project folder if needed.
# Default "." means: same folder where you run the scripts.
BASE_DIR    = os.environ.get("QSVM_BASE_DIR", ".")

PROCESSED   = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "scenarios")
SUMMARY_DIR = os.path.join(BASE_DIR, "results", "summary")

for p in [PROCESSED, RESULTS_DIR, SUMMARY_DIR]:
    os.makedirs(p, exist_ok=True)

# ── 2. EXPERIMENT PARAMETERS ─────────────────────────────────
# n_repeats=10 (5-fold × 10 repeats = 50 AUC estimates per scenario)
# Set N_ITER=1 for a quick smoke-test run.
N_ITER = 10
N_JOBS = 1          # Keep at 1; Qiskit simulation is memory-intensive
RANDOM_SEED = 42

# ── 3. FACTORIAL DESIGN ───────────────────────────────────────
N_FEATURES = [5, 10]
N_SAMPLES  = [100, 200]
IMB_RATIOS = [5, 10, 20]   # Imbalance ratio 1:5, 1:10, 1:20

# ── 4. DATA FILE PATHS ────────────────────────────────────────
FILE_POLISH_IMPUTED = os.path.join(PROCESSED, "polish_imputed.pkl")
FILE_TAIWAN_IMPUTED = os.path.join(PROCESSED, "taiwan_imputed.pkl")

# ── 5. FINANCIAL FEATURE MAPPING (Polish dataset) ────────────
# 15 harmonised financial ratios as defined in Table 2 of the paper.
# Attr37 excluded: >40% missing values.
FINANCIAL_FEATURES_POLISH = [
    "Attr1",  # Net profit / Total assets          → Net Profit Margin
    "Attr2",  # Total liabilities / Total assets   → Leverage Ratio
    "Attr3",  # Working capital / Total assets     → Working Capital / TA
    "Attr4",  # Current assets / Short-term liab.  → Current Ratio
    "Attr6",  # Retained earnings / Total assets   → Retained Earnings / TA
    "Attr7",  # EBIT / Total assets                → Return on Assets (ROA)
    "Attr8",  # Book equity / Total liabilities    → Equity / Liabilities
    "Attr9",  # Sales / Total assets               → Asset Turnover
    "Attr10", # Equity / Total assets              → Equity Ratio
    "Attr21", # Sales(n) / Sales(n-1)              → Sales Growth
    "Attr29", # log(Total assets)                  → Log Total Assets
    "Attr40", # (Cash + ST sec. - STL) / Assets    → Quick Ratio
    "Attr43", # EBIT / Sales                       → EBIT / Sales
    "Attr53", # EBITDA / Sales                     → Cash Flow Ratio
    "Attr54", # EBITDA / Total assets              → EBITDA / TA
]
