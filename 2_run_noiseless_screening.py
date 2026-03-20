import os, pickle, time, warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")
from config import *

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from joblib import Parallel, delayed

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

# ── LOAD DATA ────────────────────────────────────────────────
print("LOADING DATA...")
with open(FILE_POLISH_IMPUTED, "rb") as f: pol_data = pickle.load(f)
with open(FILE_TAIWAN_IMPUTED, "rb") as f: tai_data = pickle.load(f)

datasets = {
    "polish": (pol_data["X"], pol_data["y"]),
    "taiwan": (tai_data["X"], tai_data["y"]),
}

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def get_subset_indices(y, n_sample, imb_ratio, rng):
    min_idx = np.where(y == 1)[0]
    maj_idx = np.where(y == 0)[0]
    n_min = max(2, n_sample // (imb_ratio + 1))
    n_maj = n_sample - n_min
    n_min = min(n_min, len(min_idx))
    n_maj = min(n_maj, len(maj_idx))
    idx = np.concatenate([rng.choice(min_idx, n_min, replace=False), 
                         rng.choice(maj_idx, n_maj, replace=False)])
    rng.shuffle(idx)
    return idx

def g_mean(y_true, y_pred):
    try:
        tn, fp, fn, tp = pd.crosstab(y_true, y_pred).values.ravel()
        return np.sqrt((tp/(tp+fn)) * (tn/(tn+fp)))
    except: return 0

def run_single_scenario(dataset_name, X_full, y_full, n_feat, n_sample, imb_ratio, seed):
    rng = np.random.default_rng(seed)
    sample_idx = get_subset_indices(y_full, n_sample, imb_ratio, rng)
    X_use, y_use = X_full[sample_idx], y_full[sample_idx]
    
    if len(np.unique(y_use)) < 2: return None

    # Khởi tạo Quantum Kernel
    fm = ZZFeatureMap(feature_dimension=n_feat, reps=2, entanglement="linear")
    qk = FidelityStatevectorKernel(feature_map=fm)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=max(1, N_ITER//5), random_state=seed)
    res_metrics = {"auc_qsvm": [], "auc_xgb": []}
    
    for tr_idx, te_idx in rskf.split(X_use, y_use):
        Xtr, Xte = X_use[tr_idx], X_use[te_idx]
        ytr, yte = y_use[tr_idx], y_use[te_idx]
        
        # Scaling & Feature Selection
        scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        
        selector = SelectKBest(f_classif, k=n_feat).fit(Xtr_s, ytr)
        Xtr_f, Xte_f = selector.transform(Xtr_s), selector.transform(Xte_s)
        
        # Xử lý mất cân bằng
        try:
            Xtr_r, ytr_r = SMOTETomek(random_state=seed).fit_resample(Xtr_f, ytr)
        except:
            Xtr_r, ytr_r = Xtr_f, ytr
            
        # Baseline XGBoost
        xgb = XGBClassifier(n_estimators=100, scale_pos_weight=(ytr_r==0).sum()/(ytr_r==1).sum(), 
                            eval_metric='logloss', random_state=seed, n_jobs=1)
        xgb.fit(Xtr_r, ytr_r)
        res_metrics["auc_xgb"].append(roc_auc_score(yte, xgb.predict_proba(Xte_f)[:, 1]))

        # QSVM
        try:
            K_train = qk.evaluate(Xtr_r)
            K_test = qk.evaluate(Xte_f, Xtr_r)
            svc = SVC(kernel="precomputed", probability=True, class_weight='balanced').fit(K_train, ytr_r)
            res_metrics["auc_qsvm"].append(roc_auc_score(yte, svc.predict_proba(K_test)[:, 1]))
        except: pass

    if not res_metrics["auc_qsvm"]: return None
    
    result = {
        "dataset": dataset_name, "n_feat": n_feat, "n_sample": n_sample, "imb_ratio": imb_ratio,
        "mean_auc_qsvm": np.mean(res_metrics["auc_qsvm"]), "mean_auc_xgb": np.mean(res_metrics["auc_xgb"]),
        "delta_auc": np.mean(res_metrics["auc_qsvm"]) - np.mean(res_metrics["auc_xgb"])
    }
    
    # LƯU FILE TỨC THÌ
    fname = f"{dataset_name}_f{n_feat}_s{n_sample}_r{imb_ratio}_noiseless.pkl"
    with open(os.path.join(RESULTS_DIR, fname), "wb") as f: pickle.dump(result, f)
    print(f"  [SAVED] {fname} | Delta: {result['delta_auc']:.4f}")
    
    return result

if __name__ == "__main__":
    tasks = [(ds, data[0], data[1], nf, ns, ir, RANDOM_SEED) 
             for ds, data in datasets.items() 
             for nf, ns, ir in product(N_FEATURES, N_SAMPLES, IMB_RATIOS)]
    
    print(f"STARTING {len(tasks)} tasks...")
    Parallel(n_jobs=N_JOBS)(delayed(run_single_scenario)(*t) for t in tasks)
    print("DONE. Check results/scenarios folder.")