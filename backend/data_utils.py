"""
Dataset loading helpers and local storage utilities.
Complements model_service.py for the lab exercise (no DB).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_service import (
    DEFAULT_ENV_LABELS,
    PREDICTION_LOG_PATH,
    SATELLITE_DIR,
    SMALL_OBJECTS_DIR,
    UPLOADS_DIR,
    ensure_data_dirs,
    load_model_meta,
    save_model_meta,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_small_object_classes() -> List[Dict[str, Any]]:
    """Return [{name, count, samples: [relative paths]}]."""
    ensure_data_dirs()
    result = []
    if not SMALL_OBJECTS_DIR.exists():
        return result
    for d in sorted(SMALL_OBJECTS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        images = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        result.append(
            {
                "name": d.name,
                "count": len(images),
                "samples": [str(p.relative_to(SMALL_OBJECTS_DIR.parent.parent)) for p in images[:8]],
                "sample_paths": [str(p) for p in images[:8]],
            }
        )
    return result


def add_images_to_class(class_name: str, file_paths: List[Path]) -> Dict[str, Any]:
    """Copy uploaded image files into datasets/small_objects/{class_name}/."""
    ensure_data_dirs()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in class_name.strip())
    if not safe:
        raise ValueError("Invalid class name")
    dest_dir = SMALL_OBJECTS_DIR / safe
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for src in file_paths:
        src = Path(src)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        dest = dest_dir / f"{stamp}_{src.name}"
        shutil.copy2(src, dest)
        saved.append(str(dest))
    # Keep classes list in meta in sync
    meta = load_model_meta()
    classes = sorted(
        [d.name for d in SMALL_OBJECTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    meta["classes"] = classes
    save_model_meta(meta)
    return {"class_name": safe, "saved": saved, "count": len(list(dest_dir.glob("*")))}


def delete_class(class_name: str) -> Dict[str, Any]:
    ensure_data_dirs()
    dest = SMALL_OBJECTS_DIR / class_name
    if not dest.exists():
        raise FileNotFoundError(f"Class '{class_name}' not found")
    shutil.rmtree(dest)
    meta = load_model_meta()
    meta["classes"] = sorted(
        [d.name for d in SMALL_OBJECTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    save_model_meta(meta)
    return {"deleted": class_name}


def list_satellite_images() -> List[Dict[str, Any]]:
    ensure_data_dirs()
    images_dir = SATELLITE_DIR / "images"
    ann_dir = SATELLITE_DIR / "annotations"
    result = []
    if not images_dir.exists():
        return result
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        ann = ann_dir / f"{img.stem}.json"
        labels: List[str] = []
        if ann.exists():
            with open(ann, "r", encoding="utf-8") as f:
                labels = json.load(f).get("labels", [])
        result.append(
            {
                "filename": img.name,
                "path": str(img),
                "labels": labels,
            }
        )
    return result


def add_satellite_image(
    image_path: Path,
    labels: List[str],
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_data_dirs()
    images_dir = SATELLITE_DIR / "images"
    ann_dir = SATELLITE_DIR / "annotations"
    image_path = Path(image_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    dest_name = filename or f"{stamp}_{image_path.name}"
    dest = images_dir / dest_name
    shutil.copy2(image_path, dest)
    # Filter to known labels when possible
    meta = load_model_meta()
    known = set(meta.get("labels") or DEFAULT_ENV_LABELS)
    clean = [l for l in labels if l in known] if known else list(labels)
    ann_path = ann_dir / f"{dest.stem}.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump({"labels": clean}, f, indent=2)
    return {"filename": dest.name, "labels": clean, "path": str(dest)}


def delete_satellite_image(filename: str) -> Dict[str, Any]:
    ensure_data_dirs()
    img = SATELLITE_DIR / "images" / filename
    if not img.exists():
        raise FileNotFoundError(filename)
    ann = SATELLITE_DIR / "annotations" / f"{img.stem}.json"
    img.unlink()
    if ann.exists():
        ann.unlink()
    return {"deleted": filename}


def save_upload(file_bytes: bytes, filename: str) -> Path:
    ensure_data_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    safe = Path(filename).name
    dest = UPLOADS_DIR / f"{stamp}_{safe}"
    dest.write_bytes(file_bytes)
    return dest


def append_prediction_log(entry: Dict[str, Any]) -> None:
    ensure_data_dirs()
    if not PREDICTION_LOG_PATH.exists():
        PREDICTION_LOG_PATH.write_text("[]", encoding="utf-8")
    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
    if not isinstance(data, list):
        data = []
    entry = dict(entry)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    data.append(entry)
    with open(PREDICTION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_prediction_log() -> List[Dict[str, Any]]:
    ensure_data_dirs()
    if not PREDICTION_LOG_PATH.exists():
        return []
    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def clear_prediction_log() -> None:
    ensure_data_dirs()
    PREDICTION_LOG_PATH.write_text("[]", encoding="utf-8")
