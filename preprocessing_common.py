# ============================================================
# preprocessing_common.py
# Module DÙNG CHUNG cho 2_run_classical_baselines.py và 3_run_quantum_kernel.py.
# Chứa logic tiền xử lý chống leakage đã verify (winsorize + impute), để cả hai
# pipeline cổ điển và quantum dùng ĐÚNG CÙNG MỘT cách xử lý outlier/missing data
# trước khi rẽ nhánh sang scaler riêng (RobustScaler cho cổ điển, MinMaxScaler
# [0,π] cho quantum encoding).
#
# LỊCH SỬ SỬA LỖI (giữ lại làm tài liệu — không xóa):
#   - Phát hiện outlier cực đoan trong Taiwan (Quick_Ratio, Current_Ratio,
#     Sales_Growth có giá trị hàng tỷ) làm RobustScaler bùng nổ giá trị scaled
#     lên ~1e12, sập LogisticRegression ở một số fold (coef→0, AUC<0.3).
#     -> Sửa: thêm winsorize_fit_transform().
#   - Bug np.percentile (không bỏ qua NaN) vs np.nanpercentile: Polish có NaN
#     thật, np.percentile trả NaN cho TOÀN BỘ cột chứa NaN, khiến IterativeImputer
#     tự động loại bỏ cột đó -> Polish mất 8/15 cột âm thầm, khiến bảng ablation
#     k=10 vô tình giống bảng full15 (chỉ còn 7 cột để chọn).
#     -> Sửa: dùng np.nanpercentile.
# ============================================================
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


def winsorize_fit_transform(Xtr, Xte, lower_pct=1, upper_pct=99):
    """Clip outlier cực đoan theo percentile TÍNH TỪ TRAIN, áp dụng cho cả train/test.
    Dùng np.nanpercentile (KHÔNG np.percentile) để không phá hỏng cột có NaN."""
    lower = np.nanpercentile(Xtr, lower_pct, axis=0)
    upper = np.nanpercentile(Xtr, upper_pct, axis=0)
    Xtr_w = np.clip(Xtr, lower, upper)
    Xte_w = np.clip(Xte, lower, upper)
    return Xtr_w, Xte_w


def winsorize_and_impute(Xtr_raw, Xte_raw, seed):
    """Bước 0-1 dùng chung: Winsorize(fit train) -> Impute(fit train).
    Trả về Xtr_imp, Xte_imp — đã sạch outlier cực đoan và NaN, SẴN SÀNG để
    rẽ nhánh sang scaler riêng (RobustScaler hoặc MinMaxScaler tùy pipeline)."""
    Xtr_w, Xte_w = winsorize_fit_transform(Xtr_raw, Xte_raw)
    imputer = IterativeImputer(max_iter=10, random_state=seed)
    Xtr_imp = imputer.fit_transform(Xtr_w)
    Xte_imp = imputer.transform(Xte_w)

    # Safety check — phải luôn giữ đủ số cột gốc, không bao giờ được rơi rụng cột
    # (đây chính là bug đã phát hiện và sửa; assert này là tripwire chống tái phát)
    assert Xtr_imp.shape[1] == Xtr_raw.shape[1], (
        f"LỖI NGHIÊM TRỌNG: số cột sau impute ({Xtr_imp.shape[1]}) khác số cột gốc "
        f"({Xtr_raw.shape[1]}). Có thể IterativeImputer đã loại bỏ cột toàn-NaN — "
        f"kiểm tra lại winsorize_fit_transform có dùng nanpercentile không."
    )
    return Xtr_imp, Xte_imp
