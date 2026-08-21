"""
evaluate.py — метрики качества и скорости, как того требует задание:
mAP50, precision, recall, F1 по классам + скорость инференса.

Запуск:
    python src/evaluate.py --weights weights/logo_detector/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from config import YOLO_DATASET_DIR, OUTPUTS_DIR, IMG_SIZE, DEVICE


def evaluate_quality(weights_path: str, data_yaml: Path) -> dict:
    """Считает mAP50 / precision / recall / F1 по каждому классу через встроенный val()."""
    model = YOLO(weights_path)
    metrics = model.val(data=str(data_yaml), split="test", verbose=False)

    names = metrics.names  # {id: имя_класса}
    per_class = {}
    # box.ap_class_index сопоставляет позицию в массивах p/r/ap50 с id класса
    for idx, class_id in enumerate(metrics.box.ap_class_index):
        p = float(metrics.box.p[idx])
        r = float(metrics.box.r[idx])
        ap50 = float(metrics.box.ap50[idx])
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class[names[int(class_id)]] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mAP50": round(ap50, 4),
        }

    overall = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision_mean": round(float(metrics.box.mp), 4),
        "recall_mean": round(float(metrics.box.mr), 4),
    }
    return {"overall": overall, "per_class": per_class}


def measure_inference_speed(weights_path: str, test_images_dir: Path, imgsz: int, device,
                             n_warmup: int = 5, n_measure: int = 50) -> dict:
    """Меряет реальную скорость инференса на тестовых изображениях (не встроенный бенчмарк)."""
    model = YOLO(weights_path)
    image_paths = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"Нет изображений в {test_images_dir} для замера скорости.")

    sample = (image_paths * ((n_warmup + n_measure) // len(image_paths) + 1))[:n_warmup + n_measure]

    # Прогрев (первый инференс всегда медленнее из-за CUDA context/JIT)
    for p in sample[:n_warmup]:
        model.predict(str(p), imgsz=imgsz, device=device, verbose=False)

    start = time.perf_counter()
    for p in sample[n_warmup:]:
        model.predict(str(p), imgsz=imgsz, device=device, verbose=False)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_measure) * 1000
    return {
        "images_measured": n_measure,
        "avg_latency_ms": round(avg_ms, 2),
        "throughput_fps": round(1000 / avg_ms, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-yaml", type=Path, default=YOLO_DATASET_DIR / "data.yaml")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    quality = evaluate_quality(args.weights, args.data_yaml)
    speed = measure_inference_speed(
        args.weights,
        YOLO_DATASET_DIR / "images" / "test",
        imgsz=IMG_SIZE,
        device=args.device,
    )

    report = {"quality": quality, "speed": speed}
    print(json.dumps(report, indent=2, ensure_ascii=False))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nОтчёт сохранён в {out_path}")
