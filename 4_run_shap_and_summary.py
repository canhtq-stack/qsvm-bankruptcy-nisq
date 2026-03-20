import os, pickle, glob, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from config import *
from sklearn.model_selection import train_test_split

# Cấu hình hiển thị biểu đồ
plt.rcParams['figure.figsize'] = [12, 8]

def generate_summary():
    print("📊 ĐANG TỔNG HỢP KẾT QUẢ TỪ SCENARIOS...")
    
    # 1. Thu thập tất cả các file kết quả
    noiseless_files = glob.glob(os.path.join(RESULTS_DIR, "*_noiseless.pkl"))
    noisy_files = glob.glob(os.path.join(RESULTS_DIR, "*_noisy.pkl"))
    
    if not noiseless_files:
        print("❌ LỖI: Không tìm thấy file _noiseless.pkl nào trong thư mục scenarios!")
        return pd.DataFrame()

    results_list = []

    # 2. Đọc kết quả Noiseless
    for f in noiseless_files:
        try:
            with open(f, "rb") as pf:
                data = pickle.load(pf)
                # Tạo key để khớp với kết quả Noisy
                key = f"{data['dataset']}_{data['n_feat']}_{data['n_sample']}_{data['imb_ratio']}"
                results_list.append({
                    "Key": key,
                    "Dataset": data['dataset'].upper(),
                    "Features": data['n_feat'],
                    "Samples": data['n_sample'],
                    "Imb_Ratio": f"1:{data['imb_ratio']}",
                    "AUC_XGB": data.get('mean_auc_xgb', 0),
                    "AUC_QSVM_Clean": data.get('mean_auc_qsvm', 0),
                    "Delta_Clean": data.get('delta_auc', 0)
                })
        except Exception as e:
            print(f"Lỗi khi đọc file {f}: {e}")

    df = pd.DataFrame(results_list)

    # 3. Khớp với kết quả Noisy (nếu có)
    if noisy_files:
        noisy_data = {}
        for f in noisy_files:
            try:
                with open(f, "rb") as pf:
                    d = pickle.load(pf)
                    k = f"{d['dataset']}_{d['n_feat']}_{d['n_sample']}_{d['imb_ratio']}"
                    noisy_data[k] = (d.get('auc_q_noisy', 0), d.get('delta_auc_noisy', 0))
            except: continue
        
        df['AUC_QSVM_Noisy'] = df['Key'].map(lambda x: noisy_data.get(x, (np.nan, np.nan))[0])
        df['Delta_Noisy'] = df['AUC_QSVM_Noisy'] - df['AUC_XGB']
    else:
        df['AUC_QSVM_Noisy'] = np.nan
        df['Delta_Noisy'] = np.nan

    # 4. Lưu bảng tổng hợp
    if not df.empty:
        df = df.drop(columns=['Key']).sort_values(by=["Dataset", "Delta_Clean"], ascending=[True, False])
        if not os.path.exists(SUMMARY_DIR): os.makedirs(SUMMARY_DIR)
        summary_path = os.path.join(SUMMARY_DIR, "final_results_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"✅ Đã lưu bảng tổng hợp tại: {summary_path}")
        print("\n--- 10 KẾT QUẢ TỐT NHẤT ---")
        print(df.head(10))
    
    return df

def run_simple_shap_demo():
    print("\n🎨 ĐANG TẠO BIỂU ĐỒ SHAP GIẢI THÍCH MÔ HÌNH...")
    
    # Thử load dữ liệu Polish làm ví dụ
    try:
        with open(FILE_POLISH_IMPUTED, "rb") as f:
            data = pickle.load(f)
            X, y, features = data['X'], data['y'], data['features']
    except:
        print("⚠️ Không tìm thấy dữ liệu Polish để vẽ SHAP.")
        return

    # Lấy mẫu Stratified đảm bảo có cả 2 lớp (0 và 1)
    try:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=min(300, len(y)-1), stratify=y, random_state=RANDOM_SEED
        )
    except:
        X_sub, y_sub = X[:200], y[:200]

    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, eval_metric='logloss')
    model.fit(X_sub, y_sub)
    
    # Tính SHAP
    explainer = shap.Explainer(model, X_sub)
    shap_values = explainer(X_sub)

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sub, feature_names=features, show=False)
    
    plot_path = os.path.join(SUMMARY_DIR, "shap_feature_importance.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ SHAP tại: {plot_path}")

if __name__ == "__main__":
    # Chạy lần lượt
    final_df = generate_summary()
    
    if not final_df.empty:
        try:
            run_simple_shap_demo()
        except Exception as e:
            print(f"⚠️ Lỗi khi vẽ SHAP: {e}")
            
    print("\n🚀 TOÀN BỘ QUY TRÌNH ĐÃ HOÀN TẤT!")