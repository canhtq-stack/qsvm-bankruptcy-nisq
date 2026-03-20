# ============================================================
# 1_preprocess_and_impute.py
# Mục tiêu: Làm sạch dữ liệu + MICE Imputation + Lưu file chuẩn
# ============================================================
import pandas as pd
import numpy as np
import pickle
import warnings
import os
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Import cấu hình từ file config.py
from config import *

warnings.filterwarnings("ignore")

print("="*60)
print("BƯỚC 1: TẢI VÀ LÀM SẠCH DỮ LIỆU THÔ")
print("="*60)

# --- 1.1 Xử lý tập dữ liệu Ba Lan (Polish) ---
# Đường dẫn file CSV gốc (điều chỉnh tên file nếu cần thiết)
path_pol = os.path.join(DATA_DIR, "polish+companies+bankruptcy+data", "Polish_3year.csv")

if os.path.exists(path_pol):
    df_pol = pd.read_csv(path_pol)
    # Thay thế dấu "?" bằng NaN để máy tính hiểu là dữ liệu thiếu
    df_pol = df_pol.replace("?", np.nan)

    # Lấy danh sách các cột đặc trưng từ config
    # Lưu ý: Attr37 thường bị loại bỏ vì quá nhiều dữ liệu trống
    X_pol_raw = df_pol[FINANCIAL_FEATURES_POLISH].apply(pd.to_numeric).values
    y_pol     = df_pol["class"].values
    print(f"✅ Polish Data loaded: {X_pol_raw.shape} | Tổng NaN: {np.isnan(X_pol_raw).sum()}")
else:
    print(f"❌ Không tìm thấy file: {path_pol}")
    X_pol_raw, y_pol = None, None

# --- 1.2 Xử lý tập dữ liệu Đài Loan (Taiwan) ---
path_tai = os.path.join(DATA_DIR, "taiwanese+bankruptcy+prediction", "taiwanese.csv")

# Mapping tên cột Đài Loan sang tên ngắn gọn (để khớp với quy trình xử lý)
TAI_MAP = {
    "Bankrupt?": "Bankrupt",
    "ROA(C) before interest and depreciation before interest": "ROA_C",
    "ROA(B) before interest and depreciation after tax": "EBITDA_TA",
    "After-tax net Interest Rate": "NPM",
    "Operating Profit Rate": "Op_profit_rate",
    "Retained Earnings to Total Assets": "RE_TA",
    "Net Income to Total Assets": "NI_TA",
    "Net worth/Assets": "Equity_ratio",
    "Debt ratio %": "Debt_ratio",
    "Total Asset Turnover": "Asset_turnover",
    "Current Ratio": "Current_ratio",
    "Quick Ratio": "Quick_ratio",
    "Cash/Current Liability": "Cash_CL",
    "Total assets to GNP price": "Log_TA",
    "Total Asset Growth Rate": "Asset_growth",
    "Cash Flow to Total Assets": "CFO_TA",
}

if os.path.exists(path_tai):
    df_tai = pd.read_csv(path_tai)
    df_tai.columns = df_tai.columns.str.strip()
    df_tai = df_tai.rename(columns=TAI_MAP)
    
    # Lấy 15 đặc trưng tương ứng (loại bỏ cột mục tiêu 'Bankrupt')
    features_tai = [v for k, v in TAI_MAP.items() if v != "Bankrupt"]
    X_tai_raw = df_tai[features_tai].values
    y_tai     = df_tai["Bankrupt"].values
    print(f"✅ Taiwan Data loaded: {X_tai_raw.shape} | Tổng NaN: {np.isnan(X_tai_raw).sum()}")
else:
    print(f"❌ Không tìm thấy file: {path_tai}")
    X_tai_raw, y_tai = None, None

# --- 2. MICE IMPUTATION (Xử lý dữ liệu thiếu) ---
print("\n" + "="*60)
print("BƯỚC 2: ĐANG XỬ LÝ DỮ LIỆU THIẾU (MICE IMPUTATION)")
print("="*60)

t0 = time.time()
# Khởi tạo thuật toán MICE
mice = IterativeImputer(max_iter=10, random_state=RANDOM_SEED)

if X_pol_raw is not None:
    print("  Đang xử lý tập Polish...")
    X_pol_imp = mice.fit_transform(X_pol_raw)
else: X_pol_imp = None

if X_tai_raw is not None:
    print("  Đang xử lý tập Taiwan...")
    X_tai_imp = mice.fit_transform(X_tai_raw)
else: X_tai_imp = None

print(f"✅ Hoàn thành Imputation trong {(time.time()-t0)/60:.2f} phút")

# --- 3. LƯU KẾT QUẢ ---
print("\n" + "="*60)
print("BƯỚC 3: LƯU FILE KẾT QUẢ .PKL")
print("="*60)

if X_pol_imp is not None:
    with open(FILE_POLISH_IMPUTED, "wb") as f:
        pickle.dump({"X": X_pol_imp, "y": y_pol, "features": FINANCIAL_FEATURES_POLISH}, f)
    print(f"✅ Đã lưu: {FILE_POLISH_IMPUTED}")

if X_tai_imp is not None:
    with open(FILE_TAIWAN_IMPUTED, "wb") as f:
        pickle.dump({"X": X_tai_imp, "y": y_tai, "features": list(TAI_MAP.values())[1:]}, f)
    print(f"✅ Đã lưu: {FILE_TAIWAN_IMPUTED}")

print("\n👉 BÂY GIỜ BẠN CÓ THỂ CHẠY FILE: 2_run_noiseless_screening.py")