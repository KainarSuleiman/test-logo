"""
prepare_dataset.py — подготовка QMUL-OpenLogo / LogoDet-3K под обучение YOLO.
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

from config import TARGET_CLASSES, RAW_DATASET_DIR, YOLO_DATASET_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[Box]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_node = root.find("size")
    width = int(size_node.find("width").text)
    height = int(size_node.find("height").text)
    boxes: list[Box] = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower().replace(" ", "_")
        bnd = obj.find("bndbox")
        boxes.append(
            Box(
                class_name=name,
                x_min=float(bnd.find("xmin").text),
                y_min=float(bnd.find("ymin").text),
                x_max=float(bnd.find("xmax").text),
                y_max=float(bnd.find("ymax").text),
            )
        )
    return width, height, boxes


def voc_box_to_yolo_line(box: Box, class_id: int, img_w: int, img_h: int) -> str:
    x_center = (box.x_min + box.x_max) / 2.0 / img_w
    y_center = (box.y_min + box.y_max) / 2.0 / img_h
    w = (box.x_max - box.x_min) / img_w
    h = (box.y_max - box.y_min) / img_h
    x_center, y_center = min(max(x_center, 0.0), 1.0), min(max(y_center, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def find_annotation_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    xml_files = {p.stem: p for p in raw_dir.rglob("*.xml")}
    pairs = []
    for img_path in raw_dir.rglob("*"):
        if img_path.suffix.lower() in IMAGE_EXTS and img_path.stem in xml_files:
            pairs.append((img_path, xml_files[img_path.stem]))
    return pairs


def convert(raw_dir: Path, out_dir: Path, target_classes: list[str],
            split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
            keep_negatives: bool = True, seed: int = 42) -> None:
    random.seed(seed)
    class_to_id = {name: i for i, name in enumerate(target_classes)}

    pairs = find_annotation_pairs(raw_dir)
    if not pairs:
        raise FileNotFoundError(
            f"Не найдено пар (изображение, xml) в {raw_dir}. "
            "Убедитесь, что датасет распакован и лежит в этой папке."
        )
    print(f"Найдено {len(pairs)} размеченных изображений в {raw_dir}")

    kept_positive: list[tuple[Path, list[str]]] = []
    kept_negative: list[Path] = []

    for i, (img_path, xml_path) in enumerate(pairs, start=1):
        if i % 1000 == 0 or i == len(pairs):
            print(f"  Обработано {i}/{len(pairs)} файлов...")
        try:
            width, height, boxes = parse_voc_xml(xml_path)
        except Exception as e:
            print(f"  [пропуск] не удалось распарсить {xml_path.name}: {e}")
            continue

        yolo_lines = []
        for box in boxes:
            if box.class_name in class_to_id:
                yolo_lines.append(
                    voc_box_to_yolo_line(box, class_to_id[box.class_name], width, height)
                )

        if yolo_lines:
            kept_positive.append((img_path, yolo_lines))
        elif keep_negatives:
            kept_negative.append(img_path)

    print(f"  С целевыми классами: {len(kept_positive)}")
    print(f"  Негативных (без целевых классов, оставляем как есть): {len(kept_negative)}")

    if not kept_positive:
        raise ValueError(
            "После фильтрации по TARGET_CLASSES не осталось ни одного изображения. "
            "Проверьте, что имена классов в config.py совпадают с разметкой датасета."
        )

    random.shuffle(kept_positive)
    random.shuffle(kept_negative)
    negatives_to_use = kept_negative[: max(1, len(kept_positive) // 10)]

    all_examples = [(p, lines) for p, lines in kept_positive] + \
                    [(p, []) for p in negatives_to_use]
    random.shuffle(all_examples)

    n = len(all_examples)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    splits = {
        "train": all_examples[:n_train],
        "val": all_examples[n_train:n_train + n_val],
        "test": all_examples[n_train + n_val:],
    }

    for split_name, examples in splits.items():
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, yolo_lines in examples:
            dest_img = img_out / img_path.name
            shutil.copy2(img_path, dest_img)
            (lbl_out / (img_path.stem + ".txt")).write_text("\n".join(yolo_lines))

        print(f"  {split_name}: {len(examples)} изображений")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(target_classes)},
    }
    (out_dir / "data.yaml").write_text(yaml.dump(data_yaml, allow_unicode=True, sort_keys=False))
    print(f"\nГотово. data.yaml записан в {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=YOLO_DATASET_DIR)
    parser.add_argument("--no-negatives", action="store_true",
                         help="Не подмешивать изображения без целевых логотипов")
    args = parser.parse_args()

    convert(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        target_classes=TARGET_CLASSES,
        keep_negatives=not args.no_negatives,
    )
