"""
batch_pipeline.py — (доп. пункт задания) батчевый инференс на потоке новостей +
агрегация визуальных упоминаний сущностей по источникам и времени.

Вход: CSV со столбцами image_url, source, timestamp (ISO-формат).
Пример строки:
    image_url,source,timestamp
    https://example.com/photo1.jpg,Reuters,2026-08-19T10:00:00
    https://example.com/photo2.jpg,BBC,2026-08-19T11:30:00

Этапы:
  1. Скачать все изображения из потока.
  2. Убрать дубли по perceptual hash (одна и та же фотография часто
     публикуется разными источниками или переиспользуется — считать её
     дважды исказило бы агрегацию).
  3. Прогнать батчами через модель.
  4. Агрегировать: сколько раз каждая сущность встретилась, по каким
     источникам и в каких временных окнах (по умолчанию — по дням).

Запуск:
    python src/batch_pipeline.py --input news_stream.csv --output outputs/aggregation.csv
"""
from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path

import imagehash
import pandas as pd
import requests
from PIL import Image
from ultralytics import YOLO

from config import IMG_SIZE, MIN_DETECTION_CONF, OUTPUTS_DIR

HASH_DISTANCE_THRESHOLD = 5  # чем меньше, тем строже считаем изображения "дублями"


def download_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  [пропуск] не удалось загрузить {url}: {e}")
        return None


def deduplicate(records: list[dict]) -> list[dict]:
    """Убирает дубли изображений по perceptual hash (устойчив к небольшому
    сжатию/ресайзу — в отличие от точного хеша по байтам файла)."""
    seen_hashes: list[imagehash.ImageHash] = []
    unique_records = []

    for rec in records:
        if rec["image"] is None:
            continue
        h = imagehash.phash(rec["image"])
        is_duplicate = any((h - seen) <= HASH_DISTANCE_THRESHOLD for seen in seen_hashes)
        if not is_duplicate:
            seen_hashes.append(h)
            unique_records.append(rec)

    print(f"Дедупликация: {len(records)} -> {len(unique_records)} уникальных изображений")
    return unique_records


def run_batch_inference(records: list[dict], model_path: str, batch_size: int = 16) -> list[dict]:
    model = YOLO(model_path)
    detections_flat = []

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        images = [r["image"] for r in batch]
        results = model.predict(images, imgsz=IMG_SIZE, conf=MIN_DETECTION_CONF, verbose=False)

        for rec, result in zip(batch, results):
            for box in result.boxes:
                entity = result.names[int(box.cls[0])]
                score = float(box.conf[0])
                detections_flat.append({
                    "entity": entity,
                    "score": score,
                    "source": rec["source"],
                    "timestamp": rec["timestamp"],
                    "image_url": rec["image_url"],
                })

    print(f"Всего детекций по батчу: {len(detections_flat)}")
    return detections_flat


def aggregate(detections: list[dict], time_bucket: str = "D") -> pd.DataFrame:
    """Агрегирует упоминания сущностей по источнику и временному окну.
    time_bucket: 'D' — по дням, 'H' — по часам, 'W' — по неделям (см. pandas offset aliases).
    """
    if not detections:
        return pd.DataFrame(columns=["entity", "source", "period", "mentions", "avg_score"])

    df = pd.DataFrame(detections)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["period"] = df["timestamp"].dt.to_period(time_bucket).astype(str)

    agg = (
        df.groupby(["entity", "source", "period"])
        .agg(mentions=("entity", "count"), avg_score=("score", "mean"))
        .reset_index()
        .sort_values(["period", "mentions"], ascending=[True, False])
    )
    agg["avg_score"] = agg["avg_score"].round(4)
    return agg


def main(input_csv: Path, output_csv: Path, model_path: str, time_bucket: str):
    stream = pd.read_csv(input_csv)
    required_cols = {"image_url", "source", "timestamp"}
    if not required_cols.issubset(stream.columns):
        raise ValueError(f"Входной CSV должен содержать колонки: {required_cols}")

    print(f"Загружаю {len(stream)} изображений из потока...")
    records = []
    for _, row in stream.iterrows():
        img = download_image(row["image_url"])
        records.append({
            "image": img,
            "image_url": row["image_url"],
            "source": row["source"],
            "timestamp": row["timestamp"],
        })

    unique_records = deduplicate(records)
    detections = run_batch_inference(unique_records, model_path)
    agg_df = aggregate(detections, time_bucket=time_bucket)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(output_csv, index=False)
    print(f"\nАгрегация сохранена в {output_csv}")
    print(agg_df.head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "aggregation.csv")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--time-bucket", default="D", help="D=день, H=час, W=неделя")
    args = parser.parse_args()

    main(args.input, args.output, args.weights, args.time_bucket)
