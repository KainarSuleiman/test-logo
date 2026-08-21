"""
train.py — дообучение (fine-tune) предобученной YOLO на классах логотипов.

По заданию обучение с нуля не требуется — берём веса, предобученные на COCO,
и дообучаем только под наши 10-20 классов брендов. Это быстрее, требует меньше
данных и данных на класс, и укладывается в 8 ГБ VRAM для nano/small версий.

Запуск:
    python src/train.py
    python src/train.py --arch yolov8s.pt --epochs 80   # переопределить конфиг
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import (
    YOLO_DATASET_DIR, WEIGHTS_DIR, MODEL_ARCH, IMG_SIZE,
    BATCH_SIZE, EPOCHS, DEVICE, AUGMENTATION_PARAMS,
)


def train(arch: str, epochs: int, imgsz: int, batch: int, device):
    data_yaml = YOLO_DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} не найден. Сначала запустите prepare_dataset.py."
        )

    model = YOLO(arch)  # автоматически скачает предобученные веса при первом запуске

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(WEIGHTS_DIR),
        name="logo_detector",
        patience=15,          # ранняя остановка, если val-метрики не растут 15 эпох
        exist_ok=True,
        **AUGMENTATION_PARAMS,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nОбучение завершено. Лучшие веса: {best_weights}")
    print("Используйте этот путь в inference_service.py (переменная окружения MODEL_PATH).")
    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=MODEL_ARCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    train(args.arch, args.epochs, args.imgsz, args.batch, args.device)
