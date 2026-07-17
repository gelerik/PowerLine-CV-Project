from __future__ import annotations

import csv
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "history"
HISTORY_CSV = HISTORY_DIR / "history.csv"
ORIGINAL_IMAGES_DIR = HISTORY_DIR / "images" / "original"
PREDICTED_IMAGES_DIR = HISTORY_DIR / "images" / "predicted"
RESULTS_DIR = HISTORY_DIR / "results"

CSV_COLUMNS = [
    "id",
    "date",
    "model",
    "image_name",
    "objects_count",
    "original_path",
    "predicted_path",
    "result_path",
]


def _ensure_history_dirs() -> None:
    ORIGINAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _to_relative_path(path: Path) -> str:
    return path.relative_to(BASE_DIR).as_posix()


def _resolve_history_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _normalize_image(image: Image.Image | Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError("image must be a PIL.Image.Image instance")


def _normalize_bbox(bbox: Any) -> list[float]:
    if bbox is None:
        return []
    if isinstance(bbox, (list, tuple)):
        return [float(value) for value in bbox]
    return [float(value) for value in list(bbox)]


def _normalize_detection_object(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "class": obj.get("class") or obj.get("class_name") or obj.get("name") or "",
        "confidence": float(obj.get("confidence", obj.get("conf", 0.0))),
        "bbox": _normalize_bbox(obj.get("bbox", obj.get("box", []))),
    }


def save_detection(
    original_image: Image.Image,
    predicted_image: Image.Image,
    model_name: str,
    detected_objects: list[dict[str, Any]],
    image_name: str | None = None,
) -> str:

    _ensure_history_dirs()

    detection_id = uuid.uuid4().hex
    created_at = datetime.now().isoformat(timespec="seconds")
    safe_image_name = Path(image_name or f"{detection_id}.png").name
    original_path = ORIGINAL_IMAGES_DIR / f"{detection_id}.png"
    predicted_path = PREDICTED_IMAGES_DIR / f"{detection_id}.png"
    result_path = RESULTS_DIR / f"{detection_id}.json"

    normalized_objects = [_normalize_detection_object(obj) for obj in detected_objects]

    _normalize_image(original_image).save(original_path, format="PNG")
    _normalize_image(predicted_image).save(predicted_path, format="PNG")

    result_payload = {
        "id": detection_id,
        "date": created_at,
        "model": model_name,
        "image_name": safe_image_name,
        "objects": normalized_objects,
    }
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    row = {
        "id": detection_id,
        "date": created_at,
        "model": model_name,
        "image_name": safe_image_name,
        "objects_count": len(normalized_objects),
        "original_path": _to_relative_path(original_path),
        "predicted_path": _to_relative_path(predicted_path),
        "result_path": _to_relative_path(result_path),
    }

    file_exists = HISTORY_CSV.is_file()
    with HISTORY_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return detection_id


def load_history() -> pd.DataFrame:

    if not HISTORY_CSV.is_file():
        return pd.DataFrame(columns=CSV_COLUMNS)

    history = pd.read_csv(HISTORY_CSV)
    for column in CSV_COLUMNS:
        if column not in history.columns:
            history[column] = None

    history = history[CSV_COLUMNS]
    history["objects_count"] = pd.to_numeric(history["objects_count"], errors="coerce").fillna(0).astype(int)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    return history


def clear_history() -> None:

    if HISTORY_CSV.exists():
        HISTORY_CSV.unlink()
    if (HISTORY_DIR / "images").exists():
        shutil.rmtree(HISTORY_DIR / "images")
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)


def export_history() -> bytes:

    if not HISTORY_CSV.is_file():
        return ",".join(CSV_COLUMNS).encode("utf-8")
    return HISTORY_CSV.read_bytes()


def get_detection(detection_id: str) -> dict[str, Any]:

    result_path = RESULTS_DIR / f"{detection_id}.json"
    if not result_path.is_file():
        return {
            "id": detection_id,
            "objects": [],
        }

    return json.loads(result_path.read_text(encoding="utf-8"))


def resolve_history_path(path_value: str) -> Path:

    return _resolve_history_path(path_value)
