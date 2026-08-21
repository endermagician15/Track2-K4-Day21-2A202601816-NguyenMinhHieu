from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI(title="Income Inference API", version="1.0")

ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tải file model.joblib từ AWS S3 về máy khi server khởi động.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not ARTIFACT_BUCKET:
        print("[WARNING] ARTIFACT_BUCKET chưa được cấu hình trong biến môi trường!")
        return

    # TODO 1-4: Tạo client boto3 S3 và tải model
    s3_client = boto3.client("s3")
    print(f"Đang tải model từ s3://{ARTIFACT_BUCKET}/{MODEL_KEY}...")
    s3_client.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)
    print(f"Tải model thành công về {MODEL_PATH}")


# Tải model khi module được import
if os.path.exists(MODEL_PATH) or ARTIFACT_BUCKET:
    try:
        download_model()
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        model = None
else:
    model = None


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiểm tra sức khỏe server.
    Trả về: {"status": "ok"}
    """
    # TODO 5: Trả về dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luận chính.
    Đầu vào : JSON {"features": [f1, f2, ..., f10]}
    Đầu ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")

    # TODO 6: Kiểm tra số lượng đặc trưng (phải đúng 10 cột)
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail=f"Yêu cầu chính xác 10 đặc trưng nhân khẩu học. Nhận được {len(req.features)}.",
        )

    # TODO 7 & 8: Dự đoán và trả về nhãn tương ứng
    pred = int(model.predict([req.features])[0])
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
