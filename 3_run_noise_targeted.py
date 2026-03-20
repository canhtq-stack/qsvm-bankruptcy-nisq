import os, pickle, glob, time, warnings
import numpy as np
from joblib import Parallel, delayed
from config import *

# Tắt cảnh báo để màn hình log sạch sẽ
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

# Qiskit Core
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# ── 1. HÀM KIỂM TRA DỮ LIỆU ĐẦU VÀO ────────────────────────
def load_data():
    if not os.path.exists(FILE_POLISH_IMPUTED) or not os.path.exists(FILE_TAIWAN_IMPUTED):
        print("❌ LỖI: Không tìm thấy file dữ liệu .pkl. Hãy chạy file 1_preprocess trước.")
        return None
    with open(FILE_POLISH_IMPUTED, "rb") as f: pol = pickle.load(f)
    with open(FILE_TAIWAN_IMPUTED, "rb") as f: tai = pickle.load(f)
    return {"polish": (pol["X"], pol["y"]), "taiwan": (tai["X"], tai_data["y"] if "taiwan" in locals() else tai["y"])}

# ── 2. HÀM CHẠY NHIỄU CHI TIẾT ──────────────────────────────
def run_noisy_scenario(w, datasets):
    start_time = time.time()
    ds_name = w['dataset']
    n_feat, n_sample, imb_ratio = w['n_feat'], w['n_sample'], w['imb_ratio']
    X_full, y_full = datasets[ds_name]
    
    print(f"\n>>> ĐANG CHẠY: {ds_name} | Feats: {n_feat} | Samples: {n_sample} | Ratio: 1:{imb_ratio}")

    rng = np.random.default_rng(RANDOM_SEED)
    
    # Lấy mẫu dữ liệu (Đảm bảo tính tái lập)
    min_idx, maj_idx = np.where(y_full == 1)[0], np.where(y_full == 0)[0]
    n_min = min(max(2, n_sample // (imb_ratio + 1)), len(min_idx))
    n_maj = min(n_sample - n_min, len(maj_idx))
    idx = np.concatenate([rng.choice(min_idx, n_min, replace=False), 
                         rng.choice(maj_idx, n_maj, replace=False)])
    X_use, y_use = X_full[idx], y_full[idx]

    # Khởi tạo Kernel
    fm = ZZFeatureMap(feature_dimension=n_feat, reps=2, entanglement="linear")
    qk = FidelityQuantumKernel(feature_map=fm)

    # Chạy 1 vòng 5-fold (Nhiễu chạy rất nặng nên chỉ chạy 1 vòng để lấy kết quả sớm)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_SEED)
    auc_q, auc_x = [], []

    for i, (tr_idx, te_idx) in enumerate(rskf.split(X_use, y_use)):
        print(f"   - Fold {i+1}/5...", end=" ", flush=True)
        
        Xtr, Xte = X_use[tr_idx], X_use[te_idx]
        ytr, yte = y_use[tr_idx], y_use[te_idx]

        # Tiền xử lý
        scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        sel = SelectKBest(f_classif, k=n_feat).fit(Xtr_s, ytr)
        Xtr_f, Xte_f = sel.transform(Xtr_s), sel.transform(Xte_s)

        try:
            Xtr_r, ytr_r = SMOTETomek(random_state=RANDOM_SEED).fit_resample(Xtr_f, ytr)
        except: Xtr_r, ytr_r = Xtr_f, ytr

        # XGBoost
        xgb = XGBClassifier(n_estimators=100, scale_pos_weight=(ytr_r==0).sum()/max(1,(ytr_r==1).sum()), 
                            eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=1)
        xgb.fit(Xtr_r, ytr_r)
        auc_x.append(roc_auc_score(yte, xgb.predict_proba(Xte_f)[:, 1]))

        # QSVM Noisy (Matrix Perturbation - Siêu ổn định)
        try:
            K_train = qk.evaluate(Xtr_r)
            K_test  = qk.evaluate(Xte_f, Xtr_r)
            
            # Mô phỏng lỗi NISQ (5% depolarizing noise)
            p = 0.05 
            K_tr_n = (1-p)*K_train + p*np.eye(len(K_train))
            K_te_n = (1-p)*K_test
            
            svc = SVC(kernel="precomputed", probability=True, class_weight='balanced').fit(K_tr_n, ytr_r)
            auc_q.append(roc_auc_score(yte, svc.predict_proba(K_te_n)[:, 1]))
            print("OK")
        except Exception as e:
            print(f"Lỗi fold: {e}")

    if not auc_q: return None

    res = {
        "dataset": ds_name, "n_feat": n_feat, "n_sample": n_sample, "imb_ratio": imb_ratio,
        "auc_q_noisy": np.mean(auc_q), "auc_x_noisy": np.mean(auc_x),
        "delta_auc_noisy": np.mean(auc_q) - np.mean(auc_x)
    }
    
    # Lưu file
    fname = f"{ds_name}_f{n_feat}_s{n_sample}_r{imb_ratio}_noisy.pkl"
    with open(os.path.join(RESULTS_DIR, fname), "wb") as f: pickle.dump(res, f)
    
    duration = (time.time() - start_time)/60
    print(f"✅ XONG: {fname} | Delta Noisy: {res['delta_auc_noisy']:.4f} | Time: {duration:.1f} min")
    return res

# ── 3. THỰC THI CHÍNH ───────────────────────────────────────
if __name__ == "__main__":
    datasets = load_data()
    if datasets:
        # Tìm kịch bản chiến thắng từ Bước 2
        winners = []
        for f in glob.glob(os.path.join(RESULTS_DIR, "*_noiseless.pkl")):
            with open(f, "rb") as pf:
                r = pickle.load(pf)
                if r.get('delta_auc', 0) > 0.03: winners.append(r)
        
        if not winners:
            print("Không có kịch bản ΔAUC > 0.03. Kiểm tra lại kết quả file 2.")
        else:
            print(f"Bắt đầu chạy {len(winners)} kịch bản nhiễu...")
            # Chạy tuần tự (N_JOBS=1) để bảo vệ RAM 8GB của bạn
            for w in winners:
                run_noisy_scenario(w, datasets)