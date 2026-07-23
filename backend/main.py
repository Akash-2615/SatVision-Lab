"""
FastAPI application for Small Object Classification + Satellite Multi-Label Annotation.
Local-storage only — no DB / cloud / workers.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

import admin_auth
import data_utils
from model_service import (
    ANNOTATOR_WEIGHTS,
    CLASSIFIER_WEIGHTS,
    DEFAULT_ENV_LABELS,
    SMALL_OBJECTS_DIR,
    SATELLITE_DIR,
    TRAINING_LOG_PATH,
    UPLOADS_DIR,
    detect_and_preprocess,
    ensure_data_dirs,
    get_device,
    get_gradcam_overlay_base64,
    load_annotator,
    load_classifier,
    load_model_meta,
    metrics_from_confusion,
    predict_class,
    predict_labels,
    sliding_window_patches,
    train_annotator,
    train_classifier,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Small Object + Satellite Annotation API",
    version="1.0.0",
    description="Lab exercise: local-storage CNN classifiers with CBAM attention",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model state (loaded at startup)
STATE: Dict[str, Any] = {
    "classifier": None,
    "annotator": None,
    "classes": [],
    "labels": list(DEFAULT_ENV_LABELS),
    "device": None,
    "training": {"running": False, "job_id": None, "task": None},
}


def _reload_models() -> None:
    device = get_device()
    STATE["device"] = device
    clf, classes = load_classifier(device=device)
    ann, labels = load_annotator(device=device)
    STATE["classifier"] = clf
    STATE["annotator"] = ann
    STATE["classes"] = classes
    STATE["labels"] = labels or list(DEFAULT_ENV_LABELS)
    logger.info(
        "Models loaded — classifier=%s (%d classes), annotator=%s (%d labels), device=%s",
        clf is not None,
        len(classes),
        ann is not None,
        len(STATE["labels"]),
        device,
    )


@app.on_event("startup")
def on_startup() -> None:
    ensure_data_dirs()
    _reload_models()


# Serve uploaded / dataset images for thumbnails
@app.get("/files/{kind}/{path:path}")
def serve_file(kind: str, path: str):
    roots = {
        "uploads": UPLOADS_DIR,
        "small_objects": SMALL_OBJECTS_DIR,
        "satellite": SATELLITE_DIR / "images",
    }
    if kind not in roots:
        raise HTTPException(404, "Unknown file kind")
    root = roots[kind]
    target = (root / path).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise HTTPException(403, "Invalid path")
    if not target.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(target)


# ---------------------------------------------------------------------------
# Health & Meta
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    meta = load_model_meta()
    return {
        "status": "ok",
        "models_loaded": {
            "classifier": STATE["classifier"] is not None,
            "annotator": STATE["annotator"] is not None,
        },
        "classes": STATE["classes"] or meta.get("classes", []),
        "labels": STATE["labels"] or meta.get("labels", DEFAULT_ENV_LABELS),
        "device": str(STATE["device"] or get_device()),
    }


@app.get("/meta")
def meta():
    return load_model_meta()


# ---------------------------------------------------------------------------
# Admin auth (locks training / dataset writes)
# ---------------------------------------------------------------------------


@app.post("/admin/login")
def admin_login(password: str = Form(...)):
    if not admin_auth.verify_password(password):
        raise HTTPException(401, "Invalid admin password")
    token = admin_auth.create_session()
    return {"ok": True, "token": token, "role": "admin"}


@app.post("/admin/logout")
def admin_logout(authorization: str | None = Header(default=None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    admin_auth.revoke_session(token)
    return {"ok": True}


@app.get("/admin/me")
def admin_me(authorization: str | None = Header(default=None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    return {
        "admin": admin_auth.is_admin(token),
        "password_hint": admin_auth.password_fingerprint(),
    }


def _weight_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file": path.name, "bytes": None, "mb": None}
    size = path.stat().st_size
    return {
        "exists": True,
        "file": path.name,
        "bytes": size,
        "mb": round(size / (1024 * 1024), 2),
        "path": str(path),
    }


def _enrich_classifier_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    clf = meta.get("classifier", {}) or {}
    classes = STATE["classes"] or meta.get("classes", [])
    cm = clf.get("confusion_matrix") or clf.get("test_confusion_matrix")
    derived = metrics_from_confusion(cm, classes) if cm else {}
    return {
        "trained": bool(clf.get("trained")) and STATE["classifier"] is not None,
        "accuracy": clf.get("accuracy"),
        "test_accuracy": clf.get("test_accuracy"),
        "precision_macro": clf.get("precision_macro", derived.get("precision_macro")),
        "recall_macro": clf.get("recall_macro", derived.get("recall_macro")),
        "f1_macro": clf.get("f1_macro", derived.get("f1_macro")),
        "per_class": clf.get("per_class") or derived.get("per_class") or {},
        "loss_curve": clf.get("loss_curve", []),
        "accuracy_curve": clf.get("accuracy_curve", []),
        "confusion_matrix": cm,
        "class_list": classes,
        "last_trained": clf.get("last_trained"),
        "config": clf.get("config"),
        "architecture": clf.get("architecture") or "EfficientNet-B3 + CBAM + MLP head",
        "weights": _weight_info(CLASSIFIER_WEIGHTS),
    }


def _enrich_annotator_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    ann = meta.get("annotator", {}) or {}
    return {
        "trained": bool(ann.get("trained")) and STATE["annotator"] is not None,
        "accuracy": ann.get("accuracy"),
        "exact_match_accuracy": ann.get("exact_match_accuracy"),
        "map": ann.get("map"),
        "test_map": ann.get("test_map"),
        "hamming_loss": ann.get("hamming_loss"),
        "test_hamming_loss": ann.get("test_hamming_loss"),
        "f1_macro": ann.get("f1_macro"),
        "precision_macro": ann.get("precision_macro"),
        "recall_macro": ann.get("recall_macro"),
        "subset_f1": ann.get("subset_f1"),
        "subset_precision": ann.get("subset_precision"),
        "subset_recall": ann.get("subset_recall"),
        "f1_per_label": ann.get("f1_per_label", {}),
        "precision_per_label": ann.get("precision_per_label", {}),
        "recall_per_label": ann.get("recall_per_label", {}),
        "loss_curve": ann.get("loss_curve", []),
        "map_curve": ann.get("map_curve", []),
        "labels": STATE["labels"] or meta.get("labels", DEFAULT_ENV_LABELS),
        "threshold": meta.get("threshold", 0.5),
        "last_trained": ann.get("last_trained"),
        "config": ann.get("config"),
        "architecture": ann.get("architecture") or "EfficientNet-B3 + CBAM + MLP multi-label head",
        "weights": _weight_info(ANNOTATOR_WEIGHTS),
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _save_upload(file: UploadFile) -> Path:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    return data_utils.save_upload(data, file.filename or "upload.jpg")


def _pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _rgb_array(path: Path) -> np.ndarray:
    return np.array(_pil_rgb(path))


# ---------------------------------------------------------------------------
# Task 1 — Classify
# ---------------------------------------------------------------------------


@app.post("/classify/predict")
async def classify_predict(file: UploadFile = File(...)):
    if STATE["classifier"] is None:
        raise HTTPException(400, "Classifier not trained. Train the model first.")
    path = await _save_upload(file)
    tensor = detect_and_preprocess(path, target_size=(128, 128))
    t0 = time.perf_counter()
    result = predict_class(STATE["classifier"], tensor, class_names=STATE["classes"])
    inference_ms = (time.perf_counter() - t0) * 1000.0
    rgb = np.array(_pil_rgb(path).resize((128, 128)))
    try:
        gradcam_b64 = get_gradcam_overlay_base64(
            STATE["classifier"],
            tensor,
            rgb,
            target_layer=STATE["classifier"].target_layer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("GradCAM failed: %s", exc)
        gradcam_b64 = None

    payload = {
        "class": result["class"],
        "confidence": result["confidence"],
        "top3": result["top3"],
        "probabilities": result["probabilities"],
        "gradcam_base64": gradcam_b64,
        "image_url": f"/files/uploads/{path.name}",
        "inference_ms": round(inference_ms, 1),
        "model": "EfficientNet-B3 + CBAM",
        "input_size": "128×128",
    }
    data_utils.append_prediction_log(
        {
            "task": "classifier",
            "result": result["class"],
            "confidence": result["confidence"],
            "image": path.name,
            "top3": result["top3"],
        }
    )
    return payload


@app.post("/classify/train")
async def classify_train(
    epochs: int = Form(5),
    lr: float = Form(1e-4),
    batch_size: int = Form(8),
    image_size: int = Form(128),
    _admin: str = Depends(admin_auth.require_admin),
):
    if STATE["training"]["running"]:
        raise HTTPException(409, "Training already in progress")
    classes = data_utils.list_small_object_classes()
    if len(classes) < 2:
        raise HTTPException(400, "Need at least 2 classes with images to train")
    total_imgs = sum(c["count"] for c in classes)
    if total_imgs < 4:
        raise HTTPException(400, "Need at least 4 images total to train")

    job_id = str(uuid.uuid4())[:8]
    STATE["training"] = {"running": True, "job_id": job_id, "task": "classifier"}
    try:
        result = train_classifier(
            SMALL_OBJECTS_DIR,
            config={
                "epochs": int(epochs),
                "lr": float(lr),
                "batch_size": int(batch_size),
                "image_size": int(image_size),
                "pretrained": True,
            },
        )
        _reload_models()
        return {"status": "completed", "job_id": job_id, **result}
    except Exception as exc:  # noqa: BLE001
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Training failed: {exc}") from exc
    finally:
        STATE["training"] = {"running": False, "job_id": job_id, "task": None}


@app.get("/classify/status")
def classify_status():
    return _enrich_classifier_status(load_model_meta())


# ---------------------------------------------------------------------------
# Task 2 — Annotate
# ---------------------------------------------------------------------------


@app.post("/annotate/predict")
async def annotate_predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    if STATE["annotator"] is None:
        raise HTTPException(400, "Annotator not trained. Train the model first.")
    path = await _save_upload(file)
    tensor = detect_and_preprocess(path, target_size=(224, 224))
    t0 = time.perf_counter()
    result = predict_labels(
        STATE["annotator"],
        tensor,
        threshold=float(threshold),
        label_names=STATE["labels"],
    )
    inference_ms = (time.perf_counter() - t0) * 1000.0
    rgb = np.array(_pil_rgb(path).resize((224, 224)))
    try:
        gradcam_b64 = get_gradcam_overlay_base64(
            STATE["annotator"],
            tensor,
            rgb,
            target_layer=STATE["annotator"].target_layer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("GradCAM failed: %s", exc)
        gradcam_b64 = None

    payload = {
        "labels": result["labels"],
        "threshold": result["threshold"],
        "gradcam_base64": gradcam_b64,
        "image_url": f"/files/uploads/{path.name}",
        "inference_ms": round(inference_ms, 1),
        "model": "EfficientNet-B3 + CBAM",
        "input_size": "224×224",
    }
    above = [l["name"] for l in result["labels"] if l["above_threshold"]]
    data_utils.append_prediction_log(
        {
            "task": "annotator",
            "result": ", ".join(above) if above else "(none)",
            "confidence": max((l["score"] for l in result["labels"]), default=0.0),
            "image": path.name,
            "labels": result["labels"],
        }
    )
    return payload


@app.post("/annotate/train")
async def annotate_train(
    epochs: int = Form(5),
    lr: float = Form(3e-4),
    batch_size: int = Form(4),
    image_size: int = Form(224),
    threshold: float = Form(0.5),
    _admin: str = Depends(admin_auth.require_admin),
):
    if STATE["training"]["running"]:
        raise HTTPException(409, "Training already in progress")
    sats = data_utils.list_satellite_images()
    if len(sats) < 2:
        raise HTTPException(400, "Need at least 2 annotated satellite images")

    job_id = str(uuid.uuid4())[:8]
    STATE["training"] = {"running": True, "job_id": job_id, "task": "annotator"}
    meta = load_model_meta()
    labels = meta.get("labels") or list(DEFAULT_ENV_LABELS)
    try:
        result = train_annotator(
            SATELLITE_DIR,
            labels=labels,
            config={
                "epochs": int(epochs),
                "lr": float(lr),
                "batch_size": int(batch_size),
                "image_size": int(image_size),
                "threshold": float(threshold),
                "pretrained": True,
            },
        )
        _reload_models()
        return {"status": "completed", "job_id": job_id, **result}
    except Exception as exc:  # noqa: BLE001
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Training failed: {exc}") from exc
    finally:
        STATE["training"] = {"running": False, "job_id": job_id, "task": None}


@app.get("/annotate/status")
def annotate_status():
    return _enrich_annotator_status(load_model_meta())


# ---------------------------------------------------------------------------
# Unified Pipeline
# ---------------------------------------------------------------------------


@app.post("/pipeline/analyze")
async def pipeline_analyze(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    patch_size: int = Form(128),
    stride: int = Form(96),
    max_patches: int = Form(36),
    conf_min: float = Form(0.72),
    margin_min: float = Form(0.18),
):
    if STATE["annotator"] is None and STATE["classifier"] is None:
        raise HTTPException(400, "No models trained. Train at least one model first.")

    path = await _save_upload(file)
    rgb = _rgb_array(path)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    scene_labels: List[Dict[str, Any]] = []

    if STATE["annotator"] is not None:
        t_full = detect_and_preprocess(path, target_size=(224, 224))
        lab = predict_labels(
            STATE["annotator"],
            t_full,
            threshold=float(threshold),
            label_names=STATE["labels"],
        )
        scene_labels = lab["labels"]

    patch_results: List[Dict[str, Any]] = []
    patch_note = None
    # Object detector needs a larger scene than the 128×128 crop. Tiny EuroSAT-style
    # scenes (e.g. 64×64) get padded and force-classified as ship/vehicle/etc.
    if STATE["classifier"] is not None and (h < int(patch_size) or w < int(patch_size)):
        patch_note = (
            f"Image is {w}×{h}; object patches need at least {patch_size}×{patch_size}. "
            "Scene labels still apply — upload a larger VHR scene to detect ships/vehicles."
        )
    elif STATE["classifier"] is not None:
        patches = sliding_window_patches(rgb, patch_size=int(patch_size), stride=int(stride))
        # Cap patches for lab responsiveness
        if len(patches) > int(max_patches):
            step = max(1, len(patches) // int(max_patches))
            patches = patches[::step][: int(max_patches)]

        for p in patches:
            tensor = detect_and_preprocess(p["patch"], target_size=(128, 128))
            pred = predict_class(STATE["classifier"], tensor, class_names=STATE["classes"])
            conf = float(pred["confidence"])
            top3 = pred.get("top3") or []
            second = float(top3[1]["confidence"]) if len(top3) > 1 else 0.0
            # Require high confidence AND a clear winner (avoids water→ship guesses)
            if conf < float(conf_min) or (conf - second) < float(margin_min):
                continue
            # Encode small patch thumbnail
            pil = Image.fromarray(p["patch"])
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=70)
            thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            patch_results.append(
                {
                    "patch_id": p["patch_id"],
                    "bbox": p["bbox"],
                    "class": pred["class"],
                    "confidence": pred["confidence"],
                    "top3": pred["top3"],
                    "thumbnail_base64": thumb_b64,
                }
            )
        if not patch_results:
            patch_note = (
                "No confident small-object detections "
                f"(need ≥{int(float(conf_min)*100)}% conf and clear class margin). "
                "Scene labels above are from the multi-label annotator."
            )

    payload = {
        "scene_labels": scene_labels,
        "patch_results": patch_results,
        "patch_note": patch_note,
        "image_url": f"/files/uploads/{path.name}",
        "image_size": {"width": w, "height": h},
    }
    data_utils.append_prediction_log(
        {
            "task": "pipeline",
            "result": f"{len(patch_results)} patches",
            "confidence": max((p["confidence"] for p in patch_results), default=0.0),
            "image": path.name,
            "n_patches": len(patch_results),
            "scene_above": [l["name"] for l in scene_labels if l.get("above_threshold")],
        }
    )
    return payload


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------


@app.get("/dataset/classes")
def dataset_classes():
    return {"classes": data_utils.list_small_object_classes()}


@app.post("/dataset/classes/{class_name}")
async def dataset_add_class(
    class_name: str,
    files: List[UploadFile] = File(...),
    _admin: str = Depends(admin_auth.require_admin),
):
    tmp_paths = []
    try:
        for f in files:
            data = await f.read()
            tmp = data_utils.save_upload(data, f.filename or "img.jpg")
            tmp_paths.append(tmp)
        result = data_utils.add_images_to_class(class_name, tmp_paths)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/dataset/classes/{class_name}")
def dataset_delete_class(
    class_name: str,
    _admin: str = Depends(admin_auth.require_admin),
):
    try:
        return data_utils.delete_class(class_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/dataset/satellite")
def dataset_satellite():
    return {
        "images": data_utils.list_satellite_images(),
        "available_labels": load_model_meta().get("labels") or DEFAULT_ENV_LABELS,
    }


@app.post("/dataset/satellite")
async def dataset_add_satellite(
    file: UploadFile = File(...),
    labels: str = Form("[]"),  # JSON array string
    _admin: str = Depends(admin_auth.require_admin),
):
    try:
        label_list = json.loads(labels) if isinstance(labels, str) else list(labels)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "labels must be a JSON array") from exc
    path = await _save_upload(file)
    result = data_utils.add_satellite_image(path, label_list, filename=file.filename)
    return result


@app.delete("/dataset/satellite/{filename}")
def dataset_delete_satellite(
    filename: str,
    _admin: str = Depends(admin_auth.require_admin),
):
    try:
        return data_utils.delete_satellite_image(filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------------------
# Training logs & prediction logs
# ---------------------------------------------------------------------------


@app.get("/training/logs")
def training_logs(n: int = 80):
    ensure_data_dirs()
    if not TRAINING_LOG_PATH.exists():
        return {"lines": [], "training": STATE["training"]}
    text = TRAINING_LOG_PATH.read_text(encoding="utf-8")
    lines = text.strip().splitlines()[-n:] if text.strip() else []
    return {"lines": lines, "training": STATE["training"]}


@app.get("/logs")
def prediction_logs():
    return {"logs": data_utils.read_prediction_log()}


@app.delete("/logs")
def clear_logs():
    data_utils.clear_prediction_log()
    return {"status": "cleared"}
