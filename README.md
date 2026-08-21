# Logo Detection in News — тестовое задание

Детекция логотипов компаний/организаций на новостных изображениях.
Формальная задача: `VisualEntities(изображение) = [(сущность, box, score)]`

## 1. Обзор методов и обоснование выбора

| Подход | Точность | Скорость | Легковесность | Риск ложных срабатываний по контексту |
|---|---|---|---|---|
| Template/feature matching (SIFT/ORB) | низкая | высокая | очень лёгкий | низкий (но пропускает деформированные логотипы) |
| Faster R-CNN (двухэтапный) | высокая | низкая | тяжёлый | низкий |
| **YOLO nano/small (one-stage)** | **средняя-высокая** | **высокая** | **лёгкий** | **низкий (closed-set)** |
| RT-DETR / DETR | высокая | средняя | средний-тяжёлый | низкий |
| Open-vocabulary (Grounding DINO, YOLO-World) | средняя | средняя-низкая | тяжёлый | **высокий** — находит бренд по контексту/стилю, а не только по видимому знаку |

**Выбор: YOLO (nano или small, семейство YOLOv8/v11).**

Обоснование:
- Соответствует прямому требованию задания (лёгкий детектор, 8 ГБ VRAM, без обучения с нуля — только fine-tune).
- One-stage архитектура даёт скорость, достаточную для батчевой обработки новостного потока.
- Multi-scale head (через FPN/PANet-подобную "шею") хорошо ловит логотипы разного размера — от мелких на общем плане до крупных на баннерах.
- В отличие от open-vocabulary детекторов (Grounding DINO, YOLO-World), обучается как **closed-set** классификатор на конкретных брендах — это снижает риск "угадывания по контексту" (например, футболист в форме без видимого логотипа), что explicitly запрещено заданием.
- Faster R-CNN и RT-DETR дают сопоставимую или чуть более высокую точность, но не проходят по требованию к лёгкости и скорости инференса.

Альтернатива к рассмотрению в отчёте: если после экспериментов nano окажется недостаточно точным на мелких/деформированных логотипах — переход на `yolov8s.pt` (small), он всё ещё укладывается в 8 ГБ VRAM.

## 2. Структура проекта

```
logo-detection-news/
├── requirements.txt
├── src/
│   ├── config.py              # классы, пути, пороги — единая точка настройки
│   ├── prepare_dataset.py     # VOC-XML -> формат YOLO, train/val/test split
│   ├── train.py                # fine-tune YOLO с аугментациями
│   ├── evaluate.py             # mAP50, precision, recall, F1, скорость инференса
│   ├── inference_service.py    # FastAPI: POST /detect
│   └── batch_pipeline.py       # (бонус) батч-инференс + дедуп + агрегация
├── data/
│   ├── raw/                    # сюда распаковать QMUL-OpenLogo / LogoDet-3K
│   └── yolo_format/            # генерируется prepare_dataset.py
├── weights/                     # генерируется train.py
└── outputs/                     # отчёты и агрегации
```

## 3. Как запустить (на своей машине, где есть сеть и, желательно, GPU)

### Шаг 0 — окружение
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Шаг 1 — датасет
Скачайте и распакуйте один из датасетов в `data/raw/`:
- QMUL-OpenLogo: https://qmul-openlogo.github.io/
- LogoDet-3K: https://github.com/Wangjing1551/LogoDet-3K-Dataset

Отредактируйте `TARGET_CLASSES` в `src/config.py` под 10-20 брендов, реально
присутствующих в скачанном датасете (имена классов должны совпадать с разметкой).

```bash
python src/prepare_dataset.py
```

### Шаг 2 — обучение (fine-tune, не с нуля)
```bash
python src/train.py
# или с другими параметрами:
python src/train.py --arch yolov8s.pt --epochs 80
```
Лучшие веса появятся в `weights/logo_detector/weights/best.pt`.

### Шаг 3 — метрики качества и скорости
```bash
python src/evaluate.py --weights weights/logo_detector/weights/best.pt
```
Выведет и сохранит в `outputs/evaluation_report.json`: mAP50, precision,
recall, F1 по каждому классу + среднюю latency/FPS инференса.

### Шаг 4 — сервис инференса
```bash
export MODEL_PATH=weights/logo_detector/weights/best.pt
uvicorn src.inference_service:app --reload --port 8000
```
Проверка:
```bash
curl -X POST http://localhost:8000/detect \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/news_photo.jpg"}'

curl -X POST http://localhost:8000/detect/file -F "file=@local_photo.jpg"
```
Ответ:
```json
{
  "detections": [
    {"entity": "nike", "box": [120.5, 45.0, 210.2, 98.7], "score": 0.87, "confidence_level": "high"}
  ],
  "num_detections": 1,
  "inference_ms": 24.3
}
```

### Шаг 5 — проверка на внешних новостных изображениях
Соберите вручную 15-20 фото из реальных новостей (часть — с целевыми
логотипами, часть — намеренно без них) и прогоните через `/detect`.
Задание явно требует показать **и удачные, и ошибочные** предсказания —
сохраните такие примеры (скриншот ответа + сама картинка) отдельно в отчёт.

### Шаг 6 (бонус) — батчевый инференс и агрегация по потоку
```bash
python src/batch_pipeline.py --input news_stream.csv \
                              --weights weights/logo_detector/weights/best.pt \
                              --output outputs/aggregation.csv
```
Входной CSV: колонки `image_url, source, timestamp`. Скрипт скачивает
изображения, убирает дубли по perceptual hash (устойчив к пересжатию/ресайзу),
прогоняет батчами через модель и агрегирует упоминания брендов по источнику
и временному окну (день/час/неделя — флаг `--time-bucket`).

## 4. Метрики: как читать результат

- **mAP50** — усреднённая точность при пороге пересечения рамок (IoU) 0.5;
  главный показатель качества детекции в задании.
- **Precision / Recall / F1 по классам** — важно смотреть по каждому бренду
  отдельно: у редких классов (мало примеров в датасете) recall обычно ниже.
- **Latency / FPS** — измерено вручную на тестовых изображениях (не встроенный
  бенчмарк ultralytics), чтобы число отражало реальный сценарий инференс-сервиса.
- **low / medium / high** — дискретизация score по порогам из `config.py`
  (по умолчанию: <0.5 low, 0.5-0.75 medium, ≥0.75 high) — настраивается.

## 5. Ограничения и что можно улучшить

- Датасеты QMUL-OpenLogo / LogoDet-3K размечены не для новостных фото
  специфически — возможен domain gap (студийные/product-фото vs репортажные
  снимки). Рекомендуется дообучить/дообразить на небольшом наборе размеченных
  новостных изображений, если точность на них окажется заметно ниже val-метрик.
- Модель — closed-set: новый бренд без переобучения не найдёт. Это осознанный
  trade-off ради соответствия требованию "не определять по контексту".
