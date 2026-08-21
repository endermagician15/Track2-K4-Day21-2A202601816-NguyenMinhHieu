import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.
    Trả về điểm F1 của lớp dương (thu nhập > 50K) trên tập holdout.
    """
    # 1. Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # [Bonus 5]: Kiểm tra Data Drift / Imbalance
    pos_ratio_train = float((df_train["target"] == 1).mean())
    ref_ratio = 0.248
    drift_diff = abs(pos_ratio_train - ref_ratio)
    if drift_diff > 0.05:
        print(f"[CẢNH BÁO LỆCH LẠC DỮ LIỆU] Tỷ lệ lớp 1 trong tập train là {pos_ratio_train:.4f}, lệch {drift_diff*100:.2f}% so với tham chiếu (24.8%)!")
    else:
        print(f"[PHÂN PHỐI DỮ LIỆU] Tỷ lệ lớp 1 trong tập train: {pos_ratio_train:.4f} (Chuẩn)")

    # 2. Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        # 3. Ghi nhận siêu tham số
        mlflow.log_params(params)
        mlflow.log_metric("pos_class_ratio", pos_ratio_train)

        # 4. Huấn luyện GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 5. Dự đoán và tính metrics (Lớp dương, KHÔNG dùng average)
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # [Bonus 2]: Threshold Tuning quét ngưỡng từ 0.1 đến 0.9
        probs = model.predict_proba(X_eval)[:, 1]
        best_threshold = 0.5
        best_f1 = f1
        for th in np.arange(0.1, 0.95, 0.05):
            th_preds = (probs >= th).astype(int)
            th_f1 = float(f1_score(y_eval, th_preds))
            if th_f1 > best_f1:
                best_f1 = th_f1
                best_threshold = float(th)

        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_f1)

        # [Bonus 3]: Tính Precision, Recall và Confusion Matrix
        prec_pos = float(precision_score(y_eval, preds, zero_division=0))
        rec_pos = float(recall_score(y_eval, preds, zero_division=0))
        cm = confusion_matrix(y_eval, preds)

        # 6. Ghi nhận chỉ số vào MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_pos", prec_pos)
        mlflow.log_metric("recall_pos", rec_pos)
        mlflow.sklearn.log_model(model, "model")

        # 7. In kết quả
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f} | Precision: {prec_pos:.4f} | Recall: {rec_pos:.4f}")
        print(f"[Bonus 2] Best Threshold: {best_threshold:.2f} (F1={best_f1:.4f})")

        # 8. Lưu metrics ra outputs/report.json
        os.makedirs("outputs", exist_ok=True)
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "pos_ratio_train": pos_ratio_train,
            "best_threshold": best_threshold,
            "best_threshold_f1": best_f1,
            "precision_pos": prec_pos,
            "recall_pos": rec_pos,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        # [Bonus 3]: Lưu chi tiết vào outputs/detail.txt
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("=== CHI TIẾT ĐÁNH GIÁ MÔ HÌNH ===\n")
            f.write(f"Accuracy : {acc:.4f}\n")
            f.write(f"F1-Score : {f1:.4f}\n")
            f.write(f"Precision: {prec_pos:.4f}\n")
            f.write(f"Recall   : {rec_pos:.4f}\n\n")
            f.write(f"Confusion Matrix (TN, FP / FN, TP):\n{cm}\n")

        # 9. Lưu mô hình ra models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
