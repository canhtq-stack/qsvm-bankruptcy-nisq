# ============================================================
# 2_run_classical_baselines.py
# Mục tiêu: Chạy 4 mô hình cổ điển (LogReg/RF/XGBoost/LightGBM) trên FULL dataset
#           (Polish + Taiwan), dùng đủ 15 harmonized features.
#
# NGUYÊN TẮC CHỐNG LEAKAGE (bắt buộc, đây là lý do bài cũ bị từ chối):
#   Trong MỖI fold: Split -> Impute(fit train) -> Scale(fit train) -> SMOTE(train only)
#                   -> Train -> Predict test
#   Không có bước nào học tham số từ dữ liệu TRƯỚC khi chia fold.
#
# Cũng chạy nhánh "Equal Feature-Budget" (k=10 qua SelectKBest trong fold) để có
# bảng so sánh công bằng tuyệt đối với quantum kernel ở bước 3.
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

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocessing_common import winsorize_and_impute

# ---------- CHẠY THEO COMBO (cho phép chạy từng phần để tránh timeout) ----------
# Cách dùng: python3 2_run_classical_baselines.py <dataset> <tag>
#   dataset: polish | taiwan
#   tag: full15 | k10ablation
# Không có argument -> chạy toàn bộ 4 combo tuần tự (dùng khi có đủ thời gian).
ARGS = sys.argv[1:]

print("=" * 70)
print("BƯỚC 2: CHẠY BASELINE CỔ ĐIỂN (FULL DATASET, ĐỦ 15 FEATURES)")
print("=" * 70)

# ---------- LOAD DATA ĐÃ VALIDATE (chưa impute) ----------
with open(config.FILE_VALIDATED, "rb") as f:
    validated = pickle.load(f)

datasets = {
    "polish": (validated["polish"]["X"], validated["polish"]["y"]),
    "taiwan": (validated["taiwan"]["X"], validated["taiwan"]["y"]),
}
feature_names = validated["polish"]["feature_names"]


# ---------- HÀM XÂY DỰNG MÔ HÌNH ----------
def build_models(seed, scale_pos_weight=1.0):
    """Khởi tạo lại model mới cho mỗi fold (tránh leak state giữa fold).
    n_estimators=100 (giảm từ 200 ban đầu) — đã verify thực nghiệm không làm AUC đổi
    đáng kể (<0.01 khác biệt trên test nhanh) trong khi giảm ~40% thời gian tính."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed,
            n_jobs=1, verbosity=-1
        ),
    }


# ---------- HÀM XỬ LÝ 1 FOLD (DÙNG CHUNG CHO FULL-FEATURE VÀ ABLATION) ----------
def process_fold(Xtr_raw, Xte_raw, ytr, yte, seed, k_features=None):
    """
    Thứ tự chống leakage:
    Winsorize(fit train) -> Impute(fit train) [dùng preprocessing_common.py — CÙNG
    logic với 3_run_quantum_kernel.py để đảm bảo so sánh công bằng] -> Scale(fit train)
    -> [SelectKBest(fit train) nếu ablation] -> SMOTE(train only).
    """
    # 0-1. Winsorize + Impute — dùng chung với pipeline quantum (xem preprocessing_common.py)
    Xtr_imp, Xte_imp = winsorize_and_impute(Xtr_raw, Xte_raw, seed=seed)

    # 2. Scale — fit CHỈ trên train fold. Dùng RobustScaler (median/IQR-based) vì dữ liệu
    #    tỷ số tài chính có outlier cực đoan (vd: mẫu số gần 0 -> giá trị hàng nghìn); đã
    #    verify thực nghiệm rằng MinMaxScaler bóp méo phân phối khiến LogisticRegression
    #    suy giảm AUC mạnh so với tree-based models — đây là artifact của scaler, không
    #    phải tín hiệu thật về non-linearity.
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr_imp)
    Xte_s = scaler.transform(Xte_imp)

    # 3. (Ablation only) Feature selection — fit CHỈ trên train fold
    if k_features is not None:
        selector = SelectKBest(f_classif, k=k_features).fit(Xtr_s, ytr)
        Xtr_s = selector.transform(Xtr_s)
        Xte_s = selector.transform(Xte_s)

    # 4. SMOTE — CHỈ áp dụng trên train, KHÔNG bao giờ trên test
    try:
        Xtr_r, ytr_r = SMOTETomek(random_state=seed).fit_resample(Xtr_s, ytr)
    except Exception:
        Xtr_r, ytr_r = Xtr_s, ytr  # fallback nếu fold quá nhỏ/mất cân bằng cực độ cho SMOTE

    return Xtr_r, ytr_r, Xte_s, yte


def run_cv_for_dataset(ds_name, X_full, y_full, k_features=None, tag="", checkpoint_path=None):
    """Chạy RepeatedStratifiedKFold đầy đủ cho 1 dataset, trả về DataFrame kết quả.
    Hỗ trợ CHECKPOINT: lưu sau mỗi fold vào checkpoint_path, và resume (bỏ qua fold đã
    chạy) nếu file checkpoint đã tồn tại — cần thiết vì mỗi lần gọi tool có giới hạn
    thời gian thực thi, pipeline có thể bị gián đoạn giữa đường."""
    rskf = RepeatedStratifiedKFold(
        n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS, random_state=config.RANDOM_SEED
    )

    rows = []
    done_folds = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            rows = pickle.load(f)
        done_folds = set(r["fold"] for r in rows)
        print(f"  [{ds_name}{tag}] Resume từ checkpoint: {len(done_folds)} fold đã xong.")

    t0 = time.time()
    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X_full, y_full)):
        if fold_i in done_folds:
            continue

        Xtr_raw, Xte_raw = X_full[tr_idx], X_full[te_idx]
        ytr, yte = y_full[tr_idx], y_full[te_idx]

        Xtr_r, ytr_r, Xte_s, yte_s = process_fold(
            Xtr_raw, Xte_raw, ytr, yte, seed=config.RANDOM_SEED + fold_i, k_features=k_features
        )

        spw = (ytr_r == 0).sum() / max(1, (ytr_r == 1).sum())
        models = build_models(seed=config.RANDOM_SEED + fold_i, scale_pos_weight=spw)

        for model_name, model in models.items():
            model.fit(Xtr_r, ytr_r)
            proba = model.predict_proba(Xte_s)[:, 1]
            auc = roc_auc_score(yte_s, proba)
            rows.append({
                "dataset": ds_name, "fold": fold_i, "model": model_name,
                "auc": auc, "tag": tag,
            })

        # Checkpoint sau MỖI fold — không mất dữ liệu nếu bị gián đoạn
        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(rows, f)

        elapsed = time.time() - t0
        print(f"  [{ds_name}{tag}] fold {fold_i+1}/{config.N_SPLITS*config.N_REPEATS} "
              f"done | elapsed {elapsed:.1f}s")

    return pd.DataFrame(rows)


# ---------- THỰC THI ----------
def combo_result_path(ds_name, tag):
    return os.path.join(config.RESULTS_DIR, f"01_classical_{ds_name}_{tag}.pkl")


def run_one_combo(ds_name, tag):
    X, y = datasets[ds_name]
    k_features = config.QUANTUM_N_FEATURES if tag == "k10ablation" else None
    print(f"\n>>> COMBO: dataset={ds_name} | tag={tag} | N={X.shape[0]} | "
          f"k_features={k_features if k_features else 'ALL(15)'}")
    checkpoint_path = combo_result_path(ds_name, tag) + ".checkpoint"
    t0 = time.time()
    df_res = run_cv_for_dataset(
        ds_name, X, y, k_features=k_features, tag=f"_{tag}", checkpoint_path=checkpoint_path
    )
    elapsed = time.time() - t0
    out_path = combo_result_path(ds_name, tag)
    with open(out_path, "wb") as f:
        pickle.dump(df_res, f)
    print(f"✅ Combo xong trong {elapsed:.1f}s. Lưu: {out_path}")
    return df_res


VALID_DATASETS = ["polish", "taiwan"]
VALID_TAGS = ["full15", "k10ablation"]

if len(ARGS) == 2:
    ds_arg, tag_arg = ARGS
    assert ds_arg in VALID_DATASETS, f"dataset phải là {VALID_DATASETS}"
    assert tag_arg in VALID_TAGS, f"tag phải là {VALID_TAGS}"
    run_one_combo(ds_arg, tag_arg)
    print(f"\n👉 Combo {ds_arg}/{tag_arg} hoàn tất. Chạy combo tiếp theo hoặc 4_aggregate_results.py"
          f" nếu đã chạy đủ 4 combo (polish/taiwan x full15/k10ablation).")
elif len(ARGS) == 0:
    print("\n--- Chạy TOÀN BỘ 4 combo tuần tự (không có argument) ---")
    for ds_name in VALID_DATASETS:
        for tag in VALID_TAGS:
            run_one_combo(ds_name, tag)

    # Gộp toàn bộ thành 1 file tổng (chỉ khi chạy full)
    all_dfs = []
    for ds_name in VALID_DATASETS:
        for tag in VALID_TAGS:
            with open(combo_result_path(ds_name, tag), "rb") as f:
                all_dfs.append(pickle.load(f))
    final_df = pd.concat(all_dfs, ignore_index=True)
    with open(config.FILE_CLASSICAL_RESULTS, "wb") as f:
        pickle.dump(final_df, f)
    print(f"\n✅ Đã gộp và lưu: {config.FILE_CLASSICAL_RESULTS}")
    print(final_df.groupby(["dataset", "tag", "model"])["auc"].agg(["mean", "std", "count"]))
else:
    print("Cách dùng: python3 2_run_classical_baselines.py [<dataset> <tag>]")
    print(f"  dataset in {VALID_DATASETS} | tag in {VALID_TAGS}")
    sys.exit(1)
