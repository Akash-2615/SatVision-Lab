"""Train both models on the seeded synthetic dataset (lab defaults: few epochs)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_service import (
    DEFAULT_ENV_LABELS,
    SATELLITE_DIR,
    SMALL_OBJECTS_DIR,
    ensure_data_dirs,
    train_annotator,
    train_classifier,
)


def main() -> None:
    ensure_data_dirs()
    print("=" * 60)
    print("Training Small Object Classifier...")
    print("=" * 60)
    clf = train_classifier(
        SMALL_OBJECTS_DIR,
        config={
            "epochs": 5,
            "batch_size": 8,
            "lr": 1e-4,
            "image_size": 128,
            "pretrained": True,
        },
    )
    print("Classifier done:", {k: clf[k] for k in ("accuracy", "classes", "status")})

    print("=" * 60)
    print("Training Multi-Label Annotator...")
    print("=" * 60)
    ann = train_annotator(
        SATELLITE_DIR,
        labels=list(DEFAULT_ENV_LABELS),
        config={
            "epochs": 5,
            "batch_size": 4,
            "lr": 3e-4,
            "image_size": 224,
            "pretrained": True,
            "threshold": 0.5,
        },
    )
    print("Annotator done:", {k: ann[k] for k in ("map", "hamming_loss", "status")})
    print("ALL TRAINING COMPLETE")


if __name__ == "__main__":
    main()
