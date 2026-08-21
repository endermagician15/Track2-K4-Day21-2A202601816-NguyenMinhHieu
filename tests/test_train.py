import os
import json
import numpy as np
import pandas as pd
from src.train import train


FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Adult để sử dụng trong test.
    pytest cung cấp tmp_path là thư mục tạm thời tự động xóa sau khi test.
    """
    rng = np.random.default_rng(0)
    n = 200

    # TODO 1: Tạo mảng X có kích thước (n, 10) với giá trị [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))

    # TODO 2: Tạo mảng y nhị phân trong [0, 2)
    y = rng.integers(0, 2, size=n)

    # TODO 3: Xây dựng DataFrame, thêm cột target
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # TODO 4: Lưu 160 dòng đầu train.csv, 40 dòng cuối holdout.csv
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    # TODO 5: Trả về đường dẫn
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về số thực trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # TODO 6 & 7: Gọi train và kiểm tra float
    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path):
    """Kiểm tra file outputs/report.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    # TODO 8 & 9: Kiểm tra file tồn tại và có f1_score, accuracy
    assert os.path.exists("outputs/report.json")
    with open("outputs/report.json") as f:
        report = json.load(f)
    assert "f1_score" in report
    assert "accuracy" in report


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.joblib được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    # TODO 10: Kiểm tra file model tồn tại
    assert os.path.exists("models/model.joblib")
