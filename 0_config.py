# ============================================================
# 0_config.py
# Cấu hình trung tâm cho toàn bộ pipeline.
# Mọi file khác import từ đây — KHÔNG hardcode hằng số ở nơi khác.
# ============================================================
import os

# ---------- ĐƯỜNG DẪN ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

PATH_POLISH_RAW = os.path.join(DATA_DIR, "Polish_3year.csv")
PATH_TAIWAN_RAW = os.path.join(DATA_DIR, "taiwanese.csv")

# File trung gian (output của từng bước, input của bước sau)
FILE_VALIDATED = os.path.join(RESULTS_DIR, "00_validated_raw.pkl")
FILE_CLASSICAL_RESULTS = os.path.join(RESULTS_DIR, "01_classical_baseline_results.pkl")
FILE_QUANTUM_RESULTS = os.path.join(RESULTS_DIR, "02_quantum_kernel_results.pkl")
FILE_FOLD_INDICES = os.path.join(RESULTS_DIR, "00_fold_indices.pkl")
FILE_FINAL_SUMMARY = os.path.join(RESULTS_DIR, "03_final_summary.csv")
FILE_SHAP_SUMMARY = os.path.join(RESULTS_DIR, "04_shap_summary.pkl")

# ---------- REPRODUCIBILITY ----------
RANDOM_SEED = 42  # seed duy nhất, dùng nhất quán ở MỌI nơi có randomness

# ---------- THIẾT KẾ CROSS-VALIDATION ----------
# Áp dụng đồng nhất cho cả mô hình cổ điển và quantum, để so sánh công bằng.
N_SPLITS = 5          # 5-fold
N_REPEATS = 10         # lặp lại 10 lần -> 50 estimates/scenario (giữ tinh thần thiết kế cũ,
                       # nhưng lần này áp dụng NHẤT QUÁN cho cả quantum, không cắt xuống 1 lần)

# ---------- 15 FEATURES HARMONIZED ----------
# Single source of truth cho bộ 15 features Polish-Taiwan harmonized (kế thừa Table 2 bản
# thảo cũ, đã verify từng tên cột khớp 100% với dữ liệu thô — xem log kiểm tra trước khi viết).
# Format: (Tên hiển thị chuẩn, Polish Attr#, Taiwan column name sau khi .str.strip())
HARMONIZED_FEATURES = [
    ("ROA",                  "Attr7",  "ROA(C) before interest and depreciation before interest"),
    ("Net_Profit_Margin",    "Attr1",  "After-tax net Interest Rate"),
    ("Leverage_Ratio",       "Attr2",  "Debt ratio %"),
    ("Working_Capital_TA",   "Attr3",  "Working Capital to Total Assets"),
    ("Current_Ratio",        "Attr4",  "Current Ratio"),
    ("Retained_Earnings_TA", "Attr6",  "Retained Earnings to Total Assets"),
    ("Equity_Liabilities",   "Attr8",  "Equity to Liability"),
    ("Asset_Turnover",       "Attr9",  "Total Asset Turnover"),
    ("Equity_Ratio",         "Attr10", "Net worth/Assets"),
    ("EBIT_Sales",           "Attr43", "Operating Profit Rate"),
    ("Quick_Ratio",          "Attr40", "Quick Ratio"),
    ("EBITDA_TA",            "Attr54", "ROA(B) before interest and depreciation after tax"),
    ("Log_Total_Assets",     "Attr29", "Total assets to GNP price"),
    ("Sales_Growth",         "Attr21", "Total Asset Growth Rate"),
    ("Cash_Flow_Ratio",      "Attr53", "Cash Flow to Total Assets"),
]
# NOTE: 1_load_and_validate.py sẽ kiểm tra LẠI từng tên cột này tồn tại thật trong CSV thô
# trước khi sử dụng — không tin tưởng mù quáng vào danh sách tĩnh này.

# ---------- THAM SỐ MÔ HÌNH CỔ ĐIỂN ----------
CLASSICAL_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]

# ---------- THAM SỐ QUANTUM KERNEL ----------
# QUYẾT ĐỊNH THIẾT KẾ (đã benchmark thực nghiệm — xem log đo thời gian trước khi chốt):
# Statevector simulation cho ZZFeatureMap scale theo 2^n_qubits (không gian Hilbert), không
# tuyến tính theo n_features. Tại 15 qubits, mỗi fold (n_train=500) mất ~2-5 phút; với thiết
# kế CV đầy đủ (N_REPEATS=10, 2 dataset) tổng thời gian ước tính 3-8 giờ — không khả thi để
# lặp lại nhiều lần khi debug. Tại 10 qubits, mỗi fold ~6.5s — khả thi.
#
# Giải pháp: quantum kernel dùng SelectKBest(k=10) TRONG mỗi fold (chọn từ 15 harmonized
# features) để nằm trong giới hạn tính toán. Để loại bỏ nghi ngờ "thiết kế bất lợi cho
# quantum", pipeline tạo HAI bảng so sánh (xem 2_run_classical_baselines.py và
# 3_run_quantum_kernel.py):
#   (A) Bảng chính: cổ điển dùng ĐỦ 15 features (full power) vs quantum dùng 10 (qubit-
#       constrained) — phản ánh đúng giới hạn triển khai thực tế, nêu rõ trong Methods.
#   (B) Bảng ablation "Equal Feature-Budget": cổ điển CŨNG giới hạn SelectKBest(k=10) trong
#       fold, so sánh trực tiếp với quantum trên cùng feature space — loại bỏ hoàn toàn nghi
#       ngờ về bất công thiết kế.
QUANTUM_N_FEATURES = 10           # số feature dùng cho NHÁNH ABLATION equal-budget (xem dưới)
                                   # Quantum kernel CHÍNH dùng ĐỦ 15 features (xem ghi chú benchmark)
QUANTUM_TRAIN_SUBSET = 100         # ngưỡng train-subset TRƯỚC SMOTE cho quantum kernel.
QUANTUM_TEST_SUBSET = 100          # ngưỡng test-subset cho quantum kernel.
                                   # ĐÃ CẬP NHẬT 2 LẦN sau benchmark thực nghiệm liên tiếp:
                                   #   (1) Tại 15 qubits, K_train scale theo n² (2^15 Hilbert
                                   #       space): n=300 -> 35.4s (TRƯỚC SMOTE).
                                   #   (2) Phát hiện quan trọng: SMOTE OVERSAMPLE sau subsample
                                   #       làm train size THỰC TẾ tăng ~1.9x (vd: subsample=250
                                   #       -> sau SMOTE thành 484), khiến K_train thực đo = 65.7s
                                   #       chứ không phải ước tính ban đầu. Đồng thời test fold
                                   #       (1/5 dataset) quá lớn (1363-2100 dòng) để quantum kernel
                                   #       xử lý full — K_test scale theo n_test×n_train.
                                   # QUYẾT ĐỊNH: subsample CẢ train và test cho quantum (max_n=100
                                   # mỗi bên) -> ước tính ~44 phút tổng cho toàn pipeline.
                                   # KHÁC BIỆT CỐT LÕI với pipeline cũ (IREF, bị reject): (a) đây
                                   # là giới hạn TÍNH TOÁN đã đo thực nghiệm và định lượng minh
                                   # bạch trong Methods, không phải lựa chọn để tạo "advantage
                                   # window"; (b) mô hình CỔ ĐIỂN vẫn chạy trên FULL dataset
                                   # (10,503 / 6,819 dòng), chỉ riêng quantum bị giới hạn vì lý
                                   # do vật lý mô phỏng (2^n_qubits Hilbert space) — không subsample
                                   # toàn bộ thí nghiệm như bài cũ; (c) không có claim "quantum
                                   # advantage chỉ xuất hiện ở n nhỏ" — kết quả được báo cáo trung
                                   # thực dù thắng hay thua so với baseline full-scale.
QUANTUM_REPS = 2                  # độ sâu mạch ZZFeatureMap (theo Havlíček et al. 2019)
QUANTUM_ENTANGLEMENT = "linear"
RUN_EQUAL_BUDGET_ABLATION = True  # bật bảng (B) — cổ điển cũng giới hạn k=10 để so sánh công bằng

# ---------- IMBALANCE HANDLING ----------
SMOTE_RANDOM_STATE = RANDOM_SEED

# ---------- SONG SONG HÓA ----------
N_JOBS = 4  # điều chỉnh theo số core khả dụng; quantum kernel KHÔNG parallelize (memory-bound)

print(f"[0_config.py] Loaded. RESULTS_DIR={RESULTS_DIR}")
