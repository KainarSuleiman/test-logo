"""
inference_service.py — FastAPI-сервис для VisualEntities(image) -> [(entity, box, score)].

Реализует формальную функцию из задания:
    VisualEntities(изображение) = [(сущность, box, score)]

Эндпоинты:
    POST /detect          — JSON body {"image_url": "..."} ИЛИ multipart-загрузка файла
    GET  /health           — проверка живости сервиса

Запуск:
    export MODEL_PATH=weights/logo_detector/weights/best.pt
    uvicorn src.inference_service:app --host 0.0.0.0 --port 8000

Пример запроса (URL):
    curl -X POST http://localhost:8000/detect \
         -H "Content-Type: application/json" \
         -d '{"image_url": "https://example.com/news_photo.jpg"}'

Пример запроса (файл):
    curl -X POST http://localhost:8000/detect -F "file=@local_photo.jpg"
"""
from __future__ import annotations

import io
import os
import time
from typing import Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, HttpUrl
from ultralytics import YOLO

from config import SCORE_THRESHOLDS, MIN_DETECTION_CONF, IMG_SIZE

MODEL_PATH = os.environ.get("MODEL_PATH", "weights/logo_detector/weights/best.pt")

app = FastAPI(
    title="Logo Detection in News",
    description="Детекция логотипов компаний/организаций на новостных изображениях",
    version="1.0.0",
)

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    """Ленивая загрузка модели — один раз при первом запросе, не при импорте модуля."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


class DetectRequest(BaseModel):
    image_url: HttpUrl


class Detection(BaseModel):
    entity: str
    box: list[float]        # [x_min, y_min, x_max, y_max] в пикселях исходного изображения
    score: float
    confidence_level: str    # low / medium / high


class DetectResponse(BaseModel):
    detections: list[Detection]
    num_detections: int
    inference_ms: float


def discretize_score(score: float) -> str:
    if score >= SCORE_THRESHOLDS["high"]:
        return "high"
    if score >= SCORE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def load_image_from_url(url: str) -> Image.Image:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Не удалось загрузить изображение: {e}")
    try:
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="URL не указывает на валидное изображение")


def run_detection(image: Image.Image) -> DetectResponse:
    model = get_model()

    start = time.perf_counter()
    results = model.predict(image, imgsz=IMG_SIZE, conf=MIN_DETECTION_CONF, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections: list[Detection] = []
    result = results[0]
    for box in result.boxes:
        cls_id = int(box.cls[0])
        entity = result.names[cls_id]
        score = float(box.conf[0])
        x_min, y_min, x_max, y_max = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append(
            Detection(
                entity=entity,
                box=[x_min, y_min, x_max, y_max],
                score=round(score, 4),
                confidence_level=discretize_score(score),
            )
        )

    return DetectResponse(
        detections=detections,
        num_detections=len(detections),
        inference_ms=round(elapsed_ms, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/detect", response_model=DetectResponse)
def detect_from_url(payload: DetectRequest):
    image = load_image_from_url(str(payload.image_url))
    return run_detection(image)


@app.post("/detect/file", response_model=DetectResponse)
async def detect_from_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Загруженный файл не является валидным изображением")
    return run_detection(image)
