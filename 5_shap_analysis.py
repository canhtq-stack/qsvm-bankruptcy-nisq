# ============================================================
# 5_shap_analysis.py
# Mục tiêu: SHAP analysis trên CẢ HAI dataset (Polish + Taiwan) — khác biệt quan
# trọng so với bản cũ (chỉ làm Polish). Tính Jaccard Similarity Index (JSI) THẬT
# giữa top-features 2 nước, dùng RandomForest (TreeExplainer) làm model chính —
# ổn định nhất giữa 2 dataset, ngang ngửa LogisticRegression về AUC nhưng cho
# feature-interaction insight phong phú hơn linear coefficients.
#
# QUAN TRỌNG: SHAP được tính trên dữ liệu đã qua winsorize+impute+scale CÙNG pipeline
# với 2_run_classical_baselines.py (không phải dữ liệu thô) — để feature importance
# phản ánh đúng không gian mà model thực sự học, không bị outlier cực đoan làm méo.
# ============================================================
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [10, 8]

from importlib.machinery import SourceFileLoader
config = SourceFileLoader("config", "0_config.py").load_module()

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocessing_common import winsorize_and_impute

print("=" * 70)
print("BƯỚC 5: SHAP ANALYSIS TRÊN CẢ HAI DATASET")
print("=" * 70)

with open(config.FILE_VALIDATED, "rb") as f:
    validated = pickle.load(f)

feature_names = validated["polish"]["feature_names"]
shap_results = {}

for ds_name in ["polish", "taiwan"]:
    print(f"\n--- {ds_name.upper()} ---")
    X, y = validated[ds_name]["X"], validated[ds_name]["y"]

    # Train/test split đơn giản (không CV — SHAP analysis là minh họa feature importance
    # tổng thể, không phải phần đánh giá hiệu năng chính của paper)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )

    # CÙNG pipeline tiền xử lý với baseline cổ điển (winsorize + impute + scale)
    Xtr_imp, Xte_imp = winsorize_and_impute(Xtr_raw, Xte_raw, seed=config.RANDOM_SEED)
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr_imp)
    Xte_s = scaler.transform(Xte_imp)

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=1
    )
    model.fit(Xtr_s, ytr)

    # SHAP TreeExplainer — nhanh, chính xác cho tree-based models
    explainer = shap.TreeExplainer(model)
    # Lấy subsample test để vẽ (giữ tốc độ hợp lý, vẫn đại diện vì stratified split gốc)
    n_shap = min(500, Xte_s.shape[0])
    rng = np.random.default_rng(config.RANDOM_SEED)
    sub_idx = rng.choice(Xte_s.shape[0], n_shap, replace=False)
    X_shap = Xte_s[sub_idx]

    shap_values = explainer.shap_values(X_shap)
    # shap_values có thể là list [class0, class1] hoặc array (n, features, classes) tùy version
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 = bankrupt
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names, "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(importance_df.to_string(index=False))
    shap_results[ds_name] = {
        "importance_df": importance_df,
        "shap_values": sv,
        "X_shap": X_shap,
        "top10": set(importance_df.head(10)["feature"]),
        "top5": set(importance_df.head(5)["feature"]),
    }

    # Vẽ summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_shap, feature_names=feature_names, show=False)
    plot_path = os.path.join(config.FIGURES_DIR, f"shap_summary_{ds_name}.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Lưu biểu đồ: {plot_path}")

# ---------- JACCARD SIMILARITY INDEX (THẬT, không bịa như bản cũ) ----------
print("\n" + "=" * 70)
print("JACCARD SIMILARITY INDEX (JSI) — Cross-Dataset Feature Stability")
print("=" * 70)


def jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b)


jsi_top5 = jaccard(shap_results["polish"]["top5"], shap_results["taiwan"]["top5"])
jsi_top10 = jaccard(shap_results["polish"]["top10"], shap_results["taiwan"]["top10"])

common_top5 = shap_results["polish"]["top5"] & shap_results["taiwan"]["top5"]
common_top10 = shap_results["polish"]["top10"] & shap_results["taiwan"]["top10"]

print(f"JSI (top-5):  {jsi_top5:.3f}  | Common features: {sorted(common_top5)}")
print(f"JSI (top-10): {jsi_top10:.3f}  | Common features: {sorted(common_top10)}")

jsi_summary = {
    "jsi_top5": jsi_top5, "jsi_top10": jsi_top10,
    "common_top5": sorted(common_top5), "common_top10": sorted(common_top10),
    "polish_top10": shap_results["polish"]["importance_df"].head(10)["feature"].tolist(),
    "taiwan_top10": shap_results["taiwan"]["importance_df"].head(10)["feature"].tolist(),
}

# ---------- LƯU KẾT QUẢ ----------
output = {
    "polish_importance": shap_results["polish"]["importance_df"],
    "taiwan_importance": shap_results["taiwan"]["importance_df"],
    "jsi_summary": jsi_summary,
}
with open(config.FILE_SHAP_SUMMARY, "wb") as f:
    pickle.dump(output, f)

# Lưu CSV dễ đọc
combined_importance = pd.merge(
    shap_results["polish"]["importance_df"].rename(columns={"mean_abs_shap": "polish_shap"}),
    shap_results["taiwan"]["importance_df"].rename(columns={"mean_abs_shap": "taiwan_shap"}),
    on="feature", how="outer"
).sort_values("polish_shap", ascending=False)
csv_path = os.path.join(config.RESULTS_DIR, "05_shap_importance_combined.csv")
combined_importance.to_csv(csv_path, index=False)

print(f"\n✅ Đã lưu: {config.FILE_SHAP_SUMMARY}")
print(f"✅ Đã lưu: {csv_path}")
print("\n🚀 TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT!")
