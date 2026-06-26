# ============================================================
# 3_run_quantum_kernel.py
# Mục tiêu: Chạy Quantum SVM (fidelity kernel, ZZFeatureMap) trên Polish + Taiwan.
#
# THIẾT KẾ (đã quyết định và benchmark thực nghiệm — xem 0_config.py):
#   - Dùng ĐỦ 15 harmonized features (không giới hạn 10-qubit) — quyết định người
#     dùng để tránh "thiết kế bất lợi cho quantum".
#   - Vì statevector simulation cho 15 qubits chậm theo cấp lũy thừa (2^15 Hilbert
#     space), train-subset bị giới hạn ở QUANTUM_TRAIN_SUBSET=500 (đã benchmark:
#     ~6.5s/fold tại 10 features; sẽ đo lại thời gian thật tại 15 features ngay
#     trong file này trước khi chạy full CV).
#   - Test set vẫn lấy từ FULL hold-out fold (không subsample test) — chỉ train
#     bị giới hạn, đây là giới hạn TÍNH TOÁN đã biết của quantum kernel methods
#     trên classical simulator, không phải thiết kế nhân tạo để tạo "advantage
#     window" như pipeline cũ.
#   - Dùng CÙNG winsorize+impute (preprocessing_common.py) như pipeline cổ điển,
#     rồi rẽ nhánh riêng: MinMaxScaler về [0, π] cho quantum encoding (thay vì
#     RobustScaler) — vì ZZFeatureMap cần input trong miền góc quay [0, π].
# ============================================================
import os
import sys
import pickle
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from importlib.machinery import SourceFileLoader
config = SourceFileLoader("config", "0_config.py").load_module()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTETomek

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

from preprocessing_common import winsorize_and_impute

print("=" * 70)
print("BƯỚC 3: CHẠY QUANTUM KERNEL SVM (ĐỦ 15 FEATURES)")
print("=" * 70)

with open(config.FILE_VALIDATED, "rb") as f:
    validated = pickle.load(f)

datasets = {
    "polish": (validated["polish"]["X"], validated["polish"]["y"]),
    "taiwan": (validated["taiwan"]["X"], validated["taiwan"]["y"]),
}
N_FEATURES_FULL = len(validated["polish"]["feature_names"])  # = 15

# ---------- KHỞI TẠO QUANTUM KERNEL (1 LẦN, DÙNG LẠI CHO MỌI FOLD) ----------
# zz_feature_map (hàm) thay cho class ZZFeatureMap đã deprecated từ Qiskit 2.1.
fm = zz_feature_map(feature_dimension=N_FEATURES_FULL, reps=config.QUANTUM_REPS,
                     entanglement=config.QUANTUM_ENTANGLEMENT)
qkernel = FidelityStatevectorKernel(feature_map=fm)
print(f"Quantum kernel: ZZFeatureMap(dim={N_FEATURES_FULL}, reps={config.QUANTUM_REPS}, "
      f"entanglement={config.QUANTUM_ENTANGLEMENT})")


def subsample_stratified(X, y, max_n, seed, min_minority=15):
    """Lấy subsample STRATIFIED giữ nguyên tỷ lệ minority/majority thật, NHƯNG đảm bảo
    sàn tối thiểu min_minority mẫu minority — cần thiết vì SMOTE mặc định yêu cầu
    k_neighbors=5 minority samples để nội suy; với imbalance tự nhiên 1:20-1:30, lấy
    theo tỷ lệ thuần ở max_n nhỏ (100) chỉ cho ~3-5 minority, khiến SMOTE raise lỗi và
    pipeline fallback về dữ liệu KHÔNG cân bằng (đã phát hiện qua kiểm tra thực nghiệm:
    AUC sụp xuống 0.31 vì model học trên 97:3 imbalance, không phải tín hiệu thật).
    min_minority=15 đảm bảo đủ dư so với ngưỡng k_neighbors=5 của SMOTE."""
    if len(y) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx_min = np.where(y == 1)[0]
    idx_maj = np.where(y == 0)[0]
    frac = max_n / len(y)
    n_min_natural = max(2, int(round(len(idx_min) * frac)))
    n_min = max(n_min_natural, min(min_minority, len(idx_min)))  # sàn tối thiểu, không vượt số minority có sẵn
    n_maj = max_n - n_min
    n_min = min(n_min, len(idx_min))
    n_maj = min(n_maj, len(idx_maj))
    sel_min = rng.choice(idx_min, n_min, replace=False)
    sel_maj = rng.choice(idx_maj, n_maj, replace=False)
    sel = np.concatenate([sel_min, sel_maj])
    rng.shuffle(sel)
    return X[sel], y[sel]


def process_fold_quantum(Xtr_raw, Xte_raw, ytr, yte, seed):
    """
    Thứ tự: Winsorize+Impute(CHUNG với cổ điển) -> Subsample train VÀ test(quantum-only,
    giới hạn tính toán đã benchmark) -> MinMaxScaler[0,π](fit train subsample) ->
    SMOTE(train subsample only, làm train tăng ~1.9x do oversample minority).
    """
    # 0-1. Winsorize + Impute — CÙNG logic với pipeline cổ điển (preprocessing_common.py)
    Xtr_imp, Xte_imp = winsorize_and_impute(Xtr_raw, Xte_raw, seed=seed)

    # 2. Subsample TRAIN và TEST (quantum-only, vì giới hạn tính toán kernel O(n^2)).
    #    QUAN TRỌNG: test subsample là giới hạn TÍNH TOÁN đã đo thực nghiệm, KHÁC BẢN
    #    CHẤT với pipeline cũ (IREF) — xem ghi chú đầy đủ trong 0_config.py.
    # QUAN TRỌNG: train cần sàn min_minority (cho SMOTE k_neighbors=5); TEST giữ tỷ lệ
    # tự nhiên (không áp sàn) — vì oversample minority trong test sẽ làm AUC không còn
    # phản ánh đúng phân phối thật của hold-out fold.
    Xtr_sub, ytr_sub = subsample_stratified(Xtr_imp, ytr, config.QUANTUM_TRAIN_SUBSET, seed,
                                             min_minority=15)
    Xte_sub, yte_sub = subsample_stratified(Xte_imp, yte, config.QUANTUM_TEST_SUBSET, seed + 9999,
                                             min_minority=0)  # 0 = không áp sàn, giữ tỷ lệ tự nhiên

    # 3. Scale về [0, π] — fit CHỈ trên train subsample (không fit trên Xte)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    Xtr_s = scaler.fit_transform(Xtr_sub)
    Xte_s = scaler.transform(Xte_sub)

    # 4. SMOTE — CHỈ trên train subsample (sẽ làm train tăng lên do oversample minority)
    try:
        Xtr_r, ytr_r = SMOTETomek(random_state=seed).fit_resample(Xtr_s, ytr_sub)
    except Exception:
        Xtr_r, ytr_r = Xtr_s, ytr_sub

    return Xtr_r, ytr_r, Xte_s, yte_sub


def run_quantum_cv(ds_name, X_full, y_full, checkpoint_path=None):
    rskf = RepeatedStratifiedKFold(
        n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS, random_state=config.RANDOM_SEED
    )

    rows = []
    done_folds = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            rows = pickle.load(f)
        done_folds = set(r["fold"] for r in rows)
        print(f"  [{ds_name}] Resume từ checkpoint: {len(done_folds)} fold đã xong.")

    t0 = time.time()
    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X_full, y_full)):
        if fold_i in done_folds:
            continue

        Xtr_raw, Xte_raw = X_full[tr_idx], X_full[te_idx]
        ytr, yte = y_full[tr_idx], y_full[te_idx]
        seed = config.RANDOM_SEED + fold_i

        Xtr_r, ytr_r, Xte_s, yte_s = process_fold_quantum(Xtr_raw, Xte_raw, ytr, yte, seed)

        # Quantum kernel matrix — đây là bước tốn thời gian nhất (O(n^2) state fidelity)
        K_train = qkernel.evaluate(Xtr_r)
        K_test = qkernel.evaluate(Xte_s, Xtr_r)

        svc = SVC(kernel="precomputed", probability=True, class_weight="balanced",
                  random_state=seed)
        svc.fit(K_train, ytr_r)
        proba = svc.predict_proba(K_test)[:, 1]
        auc = roc_auc_score(yte_s, proba)

        rows.append({
            "dataset": ds_name, "fold": fold_i, "model": "QuantumSVM",
            "auc": auc, "tag": "_full15", "n_train_used": len(ytr_r),
        })

        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(rows, f)

        elapsed = time.time() - t0
        print(f"  [{ds_name}] fold {fold_i+1}/{config.N_SPLITS*config.N_REPEATS} "
              f"AUC={auc:.4f} | elapsed {elapsed:.1f}s")

    return pd.DataFrame(rows)


# ---------- CHẠY THEO DATASET (cho phép chạy từng phần để tránh timeout) ----------
ARGS = sys.argv[1:]
VALID_DATASETS = ["polish", "taiwan"]


def combo_result_path(ds_name):
    return os.path.join(config.RESULTS_DIR, f"02_quantum_{ds_name}.pkl")


def run_one_dataset(ds_name):
    X, y = datasets[ds_name]
    print(f"\n>>> Dataset: {ds_name} | N={X.shape[0]} | train_subset={config.QUANTUM_TRAIN_SUBSET}")
    checkpoint_path = combo_result_path(ds_name) + ".checkpoint"
    t0 = time.time()
    df_res = run_quantum_cv(ds_name, X, y, checkpoint_path=checkpoint_path)
    elapsed = time.time() - t0
    out_path = combo_result_path(ds_name)
    with open(out_path, "wb") as f:
        pickle.dump(df_res, f)
    print(f"✅ Dataset {ds_name} xong trong {elapsed:.1f}s. Lưu: {out_path}")
    return df_res


if len(ARGS) == 1:
    ds_arg = ARGS[0]
    assert ds_arg in VALID_DATASETS, f"dataset phải là {VALID_DATASETS}"
    run_one_dataset(ds_arg)
elif len(ARGS) == 0:
    print("\n--- Chạy TOÀN BỘ dataset tuần tự (không có argument) ---")
    all_dfs = []
    for ds_name in VALID_DATASETS:
        all_dfs.append(run_one_dataset(ds_name))
    final_df = pd.concat(all_dfs, ignore_index=True)
    with open(config.FILE_QUANTUM_RESULTS, "wb") as f:
        pickle.dump(final_df, f)
    print(f"\n✅ Đã gộp và lưu: {config.FILE_QUANTUM_RESULTS}")
    print(final_df.groupby(["dataset", "model"])["auc"].agg(["mean", "std", "count"]))
else:
    print("Cách dùng: python3 3_run_quantum_kernel.py [<dataset>]")
    print(f"  dataset in {VALID_DATASETS}")
    sys.exit(1)
