# ============================================================
# 1_load_and_validate.py
# Mục tiêu: Đọc dữ liệu thô, trích xuất 15 harmonized features, VALIDATE schema.
# KHÔNG impute, KHÔNG scale, KHÔNG feature-select ở bước này — các phép biến đổi
# học từ dữ liệu (imputation, scaling, selection) PHẢI nằm trong vòng lặp CV ở
# bước 2/3 để tránh leakage. Đây là lỗi gốc của pipeline cũ (MICE fit trên full
# data trước khi chia train/test) — bước này được thiết kế để KHÔNG lặp lại lỗi đó.
# ============================================================
import pandas as pd
import numpy as np
import pickle
import sys

sys.path.insert(0, ".")
from importlib.machinery import SourceFileLoader
config = SourceFileLoader("config", "0_config.py").load_module()

print("=" * 70)
print("BƯỚC 1: ĐỌC DỮ LIỆU THÔ + TRÍCH XUẤT 15 HARMONIZED FEATURES")
print("=" * 70)

# ---------- 1.1 Đọc Polish ----------
df_pol = pd.read_csv(config.PATH_POLISH_RAW)
print(f"\n[Polish] Shape gốc: {df_pol.shape}")

# ---------- 1.2 Đọc Taiwan ----------
df_tai = pd.read_csv(config.PATH_TAIWAN_RAW)
df_tai.columns = df_tai.columns.str.strip()  # cột Taiwan có khoảng trắng đầu tên
print(f"[Taiwan] Shape gốc: {df_tai.shape}")

# ---------- 1.3 Verify từng cột trong HARMONIZED_FEATURES tồn tại thật ----------
print("\n" + "=" * 70)
print("VALIDATE: Kiểm tra 15 harmonized features khớp dữ liệu thô")
print("=" * 70)

missing_report = []
for name, pol_col, tai_col in config.HARMONIZED_FEATURES:
    pol_ok = pol_col in df_pol.columns
    tai_ok = tai_col in df_tai.columns
    status = "OK" if (pol_ok and tai_ok) else "MISSING"
    if not pol_ok:
        missing_report.append(f"Polish column missing: {pol_col} (for {name})")
    if not tai_ok:
        missing_report.append(f"Taiwan column missing: {tai_col} (for {name})")
    print(f"  [{status:7s}] {name:22s} | Polish:{pol_col:8s} | Taiwan:'{tai_col}'")

if missing_report:
    print("\n❌ LỖI VALIDATION — DỪNG PIPELINE:")
    for m in missing_report:
        print(f"   - {m}")
    raise SystemExit("Sửa HARMONIZED_FEATURES trong 0_config.py trước khi tiếp tục.")

print("\n✅ Tất cả 15 features khớp dữ liệu thô.")

# Kiểm tra không có cột Taiwan bị dùng lại (duplicate mapping — lỗi đã phát hiện ở bản cũ)
tai_cols_used = [t for _, _, t in config.HARMONIZED_FEATURES]
from collections import Counter
dupes = {k: v for k, v in Counter(tai_cols_used).items() if v > 1}
if dupes:
    raise SystemExit(f"❌ Phát hiện Taiwan column dùng lặp lại: {dupes}. Đây là lỗi mapping, sửa trước khi tiếp tục.")
print("✅ Không có cột Taiwan bị dùng lặp lại cho 2 tỷ số khác nhau.")

# ---------- 1.4 Trích xuất feature matrix ----------
feature_names = [name for name, _, _ in config.HARMONIZED_FEATURES]
pol_cols = [pol_col for _, pol_col, _ in config.HARMONIZED_FEATURES]
tai_cols = [tai_col for _, _, tai_col in config.HARMONIZED_FEATURES]

X_pol = df_pol[pol_cols].apply(pd.to_numeric, errors="coerce").values
y_pol = df_pol["class"].astype(int).values

X_tai = df_tai[tai_cols].apply(pd.to_numeric, errors="coerce").values.copy()
y_tai = df_tai["Bankrupt?"].astype(int).values

# ---------- 1.4b SỬA LỖI ĐƠN VỊ: Log_Total_Assets ----------
# Phát hiện qua kiểm tra thực nghiệm: Polish Attr29 ("log(Total assets)") đã ở thang log
# (range: -0.36 đến 9.62), nhưng Taiwan "Total assets to GNP price" KHÔNG được log-transform
# trong dataset gốc (range: 0 đến 9.82 tỷ — 20/6819 dòng vượt 1e6). Map trực tiếp 2 cột này
# với nhau (như Table 2 bản thảo cũ làm) khiến RobustScaler bùng nổ giá trị scaled lên ~1e12
# khi gặp outlier, làm sập LogisticRegression ở một số fold (coef→0, AUC<0.3).
# Sửa: áp log1p cho riêng cột Taiwan này để đưa về thang log tương đương Polish, TRƯỚC khi
# đưa vào pipeline CV — đây là biến đổi xác định trước (deterministic, không học từ dữ liệu
# train/test), nên không gây leakage.
log_target_idx = feature_names.index("Log_Total_Assets")
print(f"\n⚠️ SỬA LỖI ĐƠN VỊ: Áp log1p cho Taiwan '{tai_cols[log_target_idx]}' "
      f"(cột gốc range 0–{X_tai[:, log_target_idx].max():.2e}, chưa log-transform trong "
      f"dataset gốc, không khớp thang với Polish Attr29 đã log-transform).")
X_tai[:, log_target_idx] = np.log1p(X_tai[:, log_target_idx])
print(f"   Sau log1p: range {X_tai[:, log_target_idx].min():.3f} đến {X_tai[:, log_target_idx].max():.3f}")

print(f"\n[Polish] X shape: {X_pol.shape} | y shape: {y_pol.shape}")
print(f"[Taiwan] X shape: {X_tai.shape} | y shape: {y_tai.shape}")

# ---------- 1.5 Báo cáo validation chi tiết (NaN, dtype, range, class balance) ----------
print("\n" + "=" * 70)
print("BÁO CÁO VALIDATION CHI TIẾT")
print("=" * 70)

validation_report = {}

for ds_name, X, y in [("polish", X_pol, y_pol), ("taiwan", X_tai, y_tai)]:
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    pct_nan_rows = (np.isnan(X).any(axis=1).sum() / X.shape[0]) * 100
    bankrupt_rate = y.mean()
    imb_ratio = (y == 0).sum() / max(1, (y == 1).sum())

    print(f"\n--- {ds_name.upper()} ---")
    print(f"  N quan sát: {X.shape[0]} | N features: {X.shape[1]}")
    print(f"  Tổng NaN: {n_nan} ({n_nan/X.size*100:.2f}% cells) | Hàng có NaN: {pct_nan_rows:.1f}%")
    print(f"  Tổng Inf: {n_inf}")
    print(f"  Bankruptcy rate: {bankrupt_rate:.2%} | Imbalance ratio: 1:{imb_ratio:.1f}")
    print(f"  NaN theo từng feature:")
    for i, fname in enumerate(feature_names):
        col_nan = np.isnan(X[:, i]).sum()
        if col_nan > 0:
            print(f"    {fname:22s}: {col_nan} NaN ({col_nan/X.shape[0]*100:.1f}%)")

    validation_report[ds_name] = {
        "n_obs": X.shape[0],
        "n_features": X.shape[1],
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
        "pct_rows_with_nan": float(pct_nan_rows),
        "bankrupt_rate": float(bankrupt_rate),
        "imbalance_ratio": float(imb_ratio),
    }

# ---------- 1.6 Cảnh báo nếu Inf tồn tại (cần xử lý trước imputation ở bước 2) ----------
if validation_report["polish"]["n_inf"] > 0 or validation_report["taiwan"]["n_inf"] > 0:
    print("\n⚠️ CẢNH BÁO: Phát hiện giá trị Inf. Bước 2/3 cần xử lý (vd: clip hoặc coerce->NaN)"
          " TRONG mỗi fold trước khi imputation.")

# ---------- 1.7 LƯU FILE (chưa impute — raw + validated) ----------
print("\n" + "=" * 70)
print("LƯU FILE: 00_validated_raw.pkl")
print("=" * 70)

output = {
    "polish": {"X": X_pol, "y": y_pol, "feature_names": feature_names},
    "taiwan": {"X": X_tai, "y": y_tai, "feature_names": feature_names},
    "validation_report": validation_report,
    "harmonized_features_map": config.HARMONIZED_FEATURES,
}

with open(config.FILE_VALIDATED, "wb") as f:
    pickle.dump(output, f)

print(f"✅ Đã lưu: {config.FILE_VALIDATED}")
print("\n⚠️ LƯU Ý QUAN TRỌNG: File này CHƯA được impute. Imputation, scaling, feature")
print("   selection và SMOTE phải được thực hiện TRONG mỗi fold CV ở bước 2 và 3,")
print("   KHÔNG được fit trên toàn bộ dữ liệu ở bước này — đây là nguyên tắc chống")
print("   leakage cốt lõi của toàn bộ pipeline.")

print("\n👉 BÂY GIỜ CHẠY: 2_run_classical_baselines.py")
