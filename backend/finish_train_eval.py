"""Finish annotator training on EuroSAT + evaluate both models on held-out test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_service import (
    DEFAULT_ENV_LABELS,
    SATELLITE_DIR,
    SatelliteDataset,
    _annotator_transforms,
    _multilabel_metrics,
    detect_and_preprocess,
    ensure_data_dirs,
    get_device,
    load_annotator,
    load_classifier,
    load_model_meta,
    predict_class,
    save_model_meta,
    train_annotator,
)

DATA = Path(__file__).resolve().parent / "data"
TEST_SO = DATA / "test" / "small_objects"
TEST_SAT = DATA / "test" / "satellite"


def eval_classifier() -> dict:
    model, classes = load_classifier()
    assert model is not None
    device = get_device()
    y_true, y_pred = [], []
    for cls in classes:
        folder = TEST_SO / cls
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.jpg")):
            t = detect_and_preprocess(p, target_size=(128, 128))
            out = predict_class(model, t, class_names=classes, device=device)
            y_true.append(cls)
            y_pred.append(out["class"])
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return {"test_accuracy": float(acc), "n": len(y_true), "confusion_matrix": cm, "classes": classes}


def eval_annotator() -> dict:
    model, labels = load_annotator()
    assert model is not None
    meta = load_model_meta()
    threshold = float(meta.get("threshold", 0.5))
    ds = SatelliteDataset(
        TEST_SAT, labels, transform=_annotator_transforms(False, 224), image_size=224
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    device = get_device()
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(targets.numpy())
    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    metrics = _multilabel_metrics(y_true, y_prob, threshold, labels)
    metrics["n"] = int(y_true.shape[0])
    return metrics


def main() -> None:
    ensure_data_dirs()
    print("Training annotator (6 epochs)…")
    ann = train_annotator(
        SATELLITE_DIR,
        labels=list(DEFAULT_ENV_LABELS),
        config={
            "epochs": 6,
            "batch_size": 8,
            "lr": 3e-4,
            "image_size": 224,
            "pretrained": True,
            "threshold": 0.5,
        },
    )
    print("annotator train:", {k: ann[k] for k in ("map", "hamming_loss", "status")})

    print("Evaluating held-out test…")
    clf_test = eval_classifier()
    ann_test = eval_annotator()
    print("classifier test:", json.dumps({k: clf_test[k] for k in ("test_accuracy", "n")}, indent=2))
    print(
        "annotator test:",
        json.dumps({k: ann_test[k] for k in ("map", "hamming_loss", "f1_macro", "n")}, indent=2),
    )

    meta = load_model_meta()
    meta["classifier"]["test_accuracy"] = clf_test.get("test_accuracy")
    meta["classifier"]["test_confusion_matrix"] = clf_test.get("confusion_matrix")
    meta["annotator"]["test_map"] = ann_test.get("map")
    meta["annotator"]["test_hamming_loss"] = ann_test.get("hamming_loss")
    meta["annotator"]["test_f1_macro"] = ann_test.get("f1_macro")
    if (DATA / "dataset_manifest.json").exists():
        meta["dataset"] = json.loads((DATA / "dataset_manifest.json").read_text())
    save_model_meta(meta)

    report = {"annotator_train": ann, "classifier_test": clf_test, "annotator_test": ann_test}
    (DATA / "test_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("ALL DONE")


if __name__ == "__main__":
    main()
