# ============================================================
# 4_aggregate_results.py
# Mục tiêu: Gộp kết quả cổ điển (2_*.py) + quantum (3_*.py) thành bảng tổng hợp,
# chạy kiểm định thống kê (Wilcoxon signed-rank + Cliff's delta), xuất CSV cuối.
#
# LƯU Ý QUAN TRỌNG VỀ THIẾT KẾ SO SÁNH (đọc trước khi diễn giải kết quả):
#   - Bảng A (MAIN): Cổ điển (full15, FULL dataset 10503/6819) vs Quantum (full15,
#     train-subsample=100, test-subsample=100). Đây là so sánh THỰC TẾ TRIỂN KHAI:
#     cổ điển dùng hết khả năng dữ liệu sẵn có, quantum bị giới hạn bởi chi phí tính
#     toán mô phỏng cổ điển (2^15 Hilbert space) — KHÔNG PHẢI lựa chọn thiết kế để
#     tạo "advantage window" như pipeline cũ (IREF, bị reject).
#   - Bảng B (RESOURCE-CONSTRAINED, đổi tên từ "Equal Feature-Budget" ban đầu): vì
#     quantum kernel cuối cùng dùng ĐỦ 15 features (không giới hạn 10-qubit như kế
#     hoạch ban đầu), bảng này SO SÁNH KHÔNG HOÀN TOÀN "công bằng" theo nghĩa feature
#     count — nó so sánh "quantum (15 feat, n=100)" với "cổ điển ablation (10 feat,
#     FULL data)". Đây là so sánh hai dạng giới hạn tài nguyên KHÁC NHAU (ít dữ liệu
#     vs ít feature), không phải cùng-điều-kiện. Phải gọi đúng tên trong Methods để
#     không gây hiểu nhầm về tính "công bằng" của so sánh.
# ============================================================
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from importlib.machinery import SourceFileLoader
config = SourceFileLoader("config", "0_config.py").load_module()

print("=" * 70)
print("BƯỚC 4: GỘP KẾT QUẢ + KIỂM ĐỊNH THỐNG KÊ")
print("=" * 70)

# ---------- LOAD KẾT QUẢ ----------
with open(config.FILE_CLASSICAL_RESULTS, "rb") as f:
    df_classical = pickle.load(f)
with open(config.FILE_QUANTUM_RESULTS, "rb") as f:
    df_quantum = pickle.load(f)

print(f"\nClassical results: {len(df_classical)} rows")
print(f"Quantum results: {len(df_quantum)} rows")

# Chuẩn hóa tag để gộp chung (classical dùng "_full15"/"_k10ablation", quantum dùng "_full15")
df_classical = df_classical.copy()
df_quantum = df_quantum.copy()
df_classical["tag"] = df_classical["tag"].str.replace("^_", "", regex=True)
df_quantum["tag"] = df_quantum["tag"].str.replace("^_", "", regex=True)

all_results = pd.concat([df_classical, df_quantum], ignore_index=True, sort=False)

# ---------- BẢNG TỔNG HỢP MÔ TẢ (Mean/Std/Min/Max theo dataset/tag/model) ----------
print("\n" + "=" * 70)
print("BẢNG TỔNG HỢP AUC (Mean/Std/Min/Max)")
print("=" * 70)
summary = all_results.groupby(["dataset", "tag", "model"])["auc"].agg(
    ["mean", "std", "min", "max", "count"]
).round(4)
print(summary)

summary_path = os.path.join(config.RESULTS_DIR, "04_auc_summary_table.csv")
summary.to_csv(summary_path)
print(f"\n✅ Lưu bảng mô tả: {summary_path}")

# ---------- KIỂM ĐỊNH THỐNG KÊ: QUANTUM vs MỖI BASELINE CỔ ĐIỂN ----------
def cliffs_delta(x, y):
    """Cliff's delta — effect size ordinal, không giả định phân phối.
    delta = (#x>y - #x<y) / (n_x * n_y)."""
    x = np.asarray(x)
    y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    greater = sum(xi > yi for xi in x for yi in y)
    less = sum(xi < yi for xi in x for yi in y)
    return (greater - less) / (n_x * n_y)


def interpret_cliffs_delta(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


print("\n" + "=" * 70)
print("KIỂM ĐỊNH THỐNG KÊ: QUANTUM vs BASELINE CỔ ĐIỂN (Bảng A — full15)")
print("=" * 70)
print("(Wilcoxon signed-rank yêu cầu paired samples theo cùng fold index — quantum và")
print(" cổ điển dùng CÙNG RepeatedStratifiedKFold(random_state=42), nên fold i của cả")
print(" hai pipeline xuất phát từ CÙNG train/test split — hợp lệ để pair theo fold.)\n")

stat_rows = []
for ds_name in ["polish", "taiwan"]:
    q_sub = all_results[(all_results.dataset == ds_name) & (all_results.model == "QuantumSVM")]
    q_sub = q_sub.sort_values("fold")
    q_auc = q_sub["auc"].values

    for model_name in config.CLASSICAL_MODELS:
        c_sub = all_results[
            (all_results.dataset == ds_name) & (all_results.tag == "full15") &
            (all_results.model == model_name)
        ].sort_values("fold")
        c_auc = c_sub["auc"].values

        if len(q_auc) != len(c_auc):
            print(f"⚠️ {ds_name}/{model_name}: số fold không khớp ({len(q_auc)} vs {len(c_auc)}), bỏ qua")
            continue

        try:
            stat, p = wilcoxon(q_auc, c_auc)
        except ValueError as e:
            stat, p = np.nan, np.nan
            print(f"⚠️ {ds_name}/{model_name}: Wilcoxon lỗi ({e})")

        delta = cliffs_delta(q_auc, c_auc)
        delta_interp = interpret_cliffs_delta(delta)
        mean_diff = q_auc.mean() - c_auc.mean()

        stat_rows.append({
            "dataset": ds_name, "comparison": f"QuantumSVM_vs_{model_name}",
            "mean_auc_quantum": round(q_auc.mean(), 4),
            "mean_auc_classical": round(c_auc.mean(), 4),
            "mean_diff": round(mean_diff, 4),
            "wilcoxon_stat": stat, "p_value": p,
            "cliffs_delta": round(delta, 4), "effect_size": delta_interp,
        })
        print(f"{ds_name:8s} | Quantum vs {model_name:20s} | "
              f"AUC_Q={q_auc.mean():.4f} AUC_C={c_auc.mean():.4f} diff={mean_diff:+.4f} | "
              f"p={p:.4f} | Cliff's δ={delta:+.3f} ({delta_interp})")

stat_df = pd.DataFrame(stat_rows)

# Bonferroni correction (8 comparisons: 2 dataset x 4 model)
n_comparisons = len(stat_df)
stat_df["p_value_bonferroni"] = (stat_df["p_value"] * n_comparisons).clip(upper=1.0)
stat_df["significant_after_correction"] = stat_df["p_value_bonferroni"] < 0.05

stat_path = os.path.join(config.RESULTS_DIR, "04_statistical_tests_main.csv")
stat_df.to_csv(stat_path, index=False)
print(f"\n✅ Lưu kiểm định thống kê (Bảng A): {stat_path}")
print(f"   (Bonferroni correction áp dụng cho {n_comparisons} so sánh)")

# ---------- BẢNG B: RESOURCE-CONSTRAINED COMPARISON (quantum vs k10ablation) ----------
print("\n" + "=" * 70)
print("BẢNG B (RESOURCE-CONSTRAINED): Quantum(15feat,n=100) vs Cổ điển(10feat,FULL data)")
print("=" * 70)
print("⚠️ ĐÂY KHÔNG PHẢI so sánh 'cùng điều kiện' — feature count VÀ sample size đều")
print("   khác nhau giữa hai nhánh. Diễn giải: hai DẠNG giới hạn tài nguyên khác nhau.\n")

stat_rows_b = []
for ds_name in ["polish", "taiwan"]:
    q_sub = all_results[(all_results.dataset == ds_name) & (all_results.model == "QuantumSVM")]
    q_sub = q_sub.sort_values("fold")
    q_auc = q_sub["auc"].values

    for model_name in config.CLASSICAL_MODELS:
        c_sub = all_results[
            (all_results.dataset == ds_name) & (all_results.tag == "k10ablation") &
            (all_results.model == model_name)
        ].sort_values("fold")
        c_auc = c_sub["auc"].values
        if len(q_auc) != len(c_auc):
            continue
        try:
            stat, p = wilcoxon(q_auc, c_auc)
        except ValueError:
            stat, p = np.nan, np.nan
        delta = cliffs_delta(q_auc, c_auc)
        stat_rows_b.append({
            "dataset": ds_name, "comparison": f"QuantumSVM_vs_{model_name}_k10",
            "mean_auc_quantum": round(q_auc.mean(), 4),
            "mean_auc_classical_k10": round(c_auc.mean(), 4),
            "p_value": p, "cliffs_delta": round(delta, 4),
            "effect_size": interpret_cliffs_delta(delta),
        })

stat_df_b = pd.DataFrame(stat_rows_b)
stat_path_b = os.path.join(config.RESULTS_DIR, "04_statistical_tests_resource_constrained.csv")
stat_df_b.to_csv(stat_path_b, index=False)
print(stat_df_b.to_string(index=False))
print(f"\n✅ Lưu kiểm định thống kê (Bảng B): {stat_path_b}")

# ---------- LƯU FILE TỔNG HỢP CUỐI CÙNG ----------
with open(config.FILE_FINAL_SUMMARY.replace(".csv", "_raw.pkl"), "wb") as f:
    pickle.dump(all_results, f)
all_results.to_csv(config.FILE_FINAL_SUMMARY, index=False)
print(f"\n✅ Đã lưu file kết quả thô đầy đủ: {config.FILE_FINAL_SUMMARY}")

print("\n👉 BÂY GIỜ CHẠY: 5_shap_analysis.py")
