"""
Core deep learning module for:
  Task 1 — Small Object Classifier (EfficientNet-B3 + CBAM)
  Task 2 — Multi-Label Satellite Annotator (EfficientNet-B3 + sigmoid head)

Standalone-testable; no FastAPI dependency.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parent
DATA_ROOT = BACKEND_ROOT / "data"
MODELS_DIR = DATA_ROOT / "models"
DATASETS_DIR = DATA_ROOT / "datasets"
SMALL_OBJECTS_DIR = DATASETS_DIR / "small_objects"
SATELLITE_DIR = DATASETS_DIR / "satellite"
UPLOADS_DIR = DATA_ROOT / "uploads"
LOGS_DIR = DATA_ROOT / "logs"

CLASSIFIER_WEIGHTS = MODELS_DIR / "small_object_classifier.pth"
ANNOTATOR_WEIGHTS = MODELS_DIR / "multilabel_annotator.pth"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"
PREDICTION_LOG_PATH = LOGS_DIR / "prediction_log.json"
TRAINING_LOG_PATH = LOGS_DIR / "training_log.txt"

DEFAULT_ENV_LABELS = [
    "water_body",
    "urban_area",
    "dense_forest",
    "sparse_vegetation",
    "agriculture",
    "barren_land",
    "desert",
    "cloud_cover",
    "flood",
    "wildfire",
    "snow_ice",
    "wetland",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_data_dirs() -> None:
    """Create the local storage tree if it does not exist."""
    for d in (
        MODELS_DIR,
        SMALL_OBJECTS_DIR,
        SATELLITE_DIR / "images",
        SATELLITE_DIR / "annotations",
        UPLOADS_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

    if not MODEL_META_PATH.exists():
        meta = default_model_meta()
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    if not PREDICTION_LOG_PATH.exists():
        PREDICTION_LOG_PATH.write_text("[]", encoding="utf-8")
    if not TRAINING_LOG_PATH.exists():
        TRAINING_LOG_PATH.write_text("", encoding="utf-8")


def default_model_meta() -> Dict[str, Any]:
    return {
        "classes": [],
        "labels": list(DEFAULT_ENV_LABELS),
        "threshold": 0.5,
        "classifier": {
            "trained": False,
            "accuracy": None,
            "best_val_accuracy": None,
            "loss_curve": [],
            "accuracy_curve": [],
            "confusion_matrix": None,
            "last_trained": None,
        },
        "annotator": {
            "trained": False,
            "map": None,
            "hamming_loss": None,
            "f1_per_label": {},
            "loss_curve": [],
            "map_curve": [],
            "last_trained": None,
        },
        "updated_at": None,
    }


def load_model_meta() -> Dict[str, Any]:
    ensure_data_dirs()
    with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_meta(meta: Dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def append_training_log(message: str) -> None:
    ensure_data_dirs()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


# ---------------------------------------------------------------------------
# CBAM Attention (Channel + Spatial)
# ---------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    """Channel attention: avg/max pool → shared MLP → sigmoid gate."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise avg/max → Conv2d(1) → sigmoid gate."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


class SmallObjectClassifier(nn.Module):
    """
    EfficientNet-B3 (features) → CBAM → GAP → MLP head → logits
    Input: 3 x H x W (typically 128).
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        feat_channels = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(feat_channels, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(feat_channels // 2, 256)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_channels, hidden),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.25),
            nn.Linear(hidden, num_classes),
        )
        self.num_classes = num_classes
        self._gradcam_activations: Optional[torch.Tensor] = None
        self.cbam.register_forward_hook(self._save_activation)

    def _save_activation(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self._gradcam_activations = out

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]
        feats = self.cbam(feats)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        x = self.pool(feats).flatten(1)
        return self.head(x)

    @property
    def target_layer(self) -> nn.Module:
        return self.cbam


class MultiLabelAnnotator(nn.Module):
    """
    EfficientNet-B3 → CBAM → GAP → MLP → multi-label logits
    Raw logits; sigmoid at inference / BCEWithLogitsLoss in training.
    Input: 3 x 224 x 224.
    """

    def __init__(self, num_labels: int, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.num_features = self.backbone.num_features
        self.cbam = CBAM(self.num_features, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(self.num_features // 2, 256)
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.num_features, hidden),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels),
        )
        self.num_labels = num_labels
        self._gradcam_activations: Optional[torch.Tensor] = None
        self.cbam.register_forward_hook(self._save_activation)

    def _save_activation(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self._gradcam_activations = out

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        return self.cbam(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        x = self.pool(feats).flatten(1)
        return self.head(x)

    @property
    def target_layer(self) -> nn.Module:
        return self.cbam


def build_small_object_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    return SmallObjectClassifier(num_classes=num_classes, pretrained=pretrained)


def build_multilabel_annotator(num_labels: int, pretrained: bool = True) -> nn.Module:
    return MultiLabelAnnotator(num_labels=num_labels, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def detect_and_preprocess(
    image_path: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """
    Load an image from path / PIL / ndarray and return a normalized
    CHW float tensor of shape (1, 3, H, W).
    """
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    elif isinstance(image_path, np.ndarray):
        if image_path.ndim == 2:
            image_path = cv2.cvtColor(image_path, cv2.COLOR_GRAY2RGB)
        elif image_path.shape[2] == 4:
            image_path = cv2.cvtColor(image_path, cv2.COLOR_BGRA2RGB)
        elif image_path.shape[2] == 3:
            # Assume BGR from OpenCV unless already RGB-looking; convert BGR→RGB
            image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_path.astype(np.uint8))
    else:
        img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def tensor_from_bytes(
    data: bytes,
    target_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return detect_and_preprocess(img, target_size=target_size)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SatelliteDataset(Dataset):
    """
    Reads satellite images from images/ and multi-hot labels from
    matching JSON files in annotations/ ({"labels": ["water_body", ...]}).
    """

    def __init__(
        self,
        root: Union[str, Path],
        label_names: List[str],
        transform: Optional[Any] = None,
        image_size: int = 224,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.label_names = list(label_names)
        self.label_to_idx = {n: i for i, n in enumerate(self.label_names)}
        self.transform = transform
        self.image_size = image_size

        self.samples: List[Tuple[Path, np.ndarray]] = []
        if self.images_dir.exists():
            for img_path in sorted(self.images_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
                    continue
                ann_path = self.ann_dir / f"{img_path.stem}.json"
                labels = []
                if ann_path.exists():
                    with open(ann_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    labels = data.get("labels", [])
                multi_hot = np.zeros(len(self.label_names), dtype=np.float32)
                for lab in labels:
                    if lab in self.label_to_idx:
                        multi_hot[self.label_to_idx[lab]] = 1.0
                self.samples.append((img_path, multi_hot))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, multi_hot = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Albumentations ToTensorV2 yields CHW tensor already
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        target = torch.from_numpy(multi_hot)
        return image, target


def _classifier_transforms(train: bool, image_size: int = 128):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        # Fallback without albumentations
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _annotator_transforms(train: bool, image_size: int = 224):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        return None

    if train:
        return A.Compose(
            [
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.CLAHE(p=0.3),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class AlbumentationsImageFolder(Dataset):
    """ImageFolder-compatible dataset using albumentations transforms."""

    def __init__(self, root: Union[str, Path], transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            for p in (self.root / cls).iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                    self.samples.append((p, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            # Albumentations
            if hasattr(self.transform, "__call__") and not isinstance(self.transform, transforms.Compose):
                try:
                    image = self.transform(image=image)["image"]
                except TypeError:
                    image = self.transform(Image.fromarray(image))
            else:
                image = self.transform(Image.fromarray(image))
        else:
            image = transforms.ToTensor()(Image.fromarray(image))
        return image, label


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds == targets).float().mean().item()


def metrics_from_confusion(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Derive accuracy / precision / recall / F1 from a confusion matrix."""
    import numpy as np

    mat = np.array(cm, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "per_class": {},
        }
    n = mat.shape[0]
    names = class_names or [f"class_{i}" for i in range(n)]
    total = mat.sum()
    accuracy = float(np.trace(mat) / total) if total > 0 else 0.0
    per_class: Dict[str, Dict[str, float]] = {}
    precs, recalls, f1s = [], [], []
    for i in range(n):
        tp = mat[i, i]
        fp = mat[:, i].sum() - tp
        fn = mat[i, :].sum() - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        name = names[i] if i < len(names) else f"class_{i}"
        per_class[name] = {"precision": prec, "recall": rec, "f1": f1, "support": int(mat[i, :].sum())}
        precs.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precs)) if precs else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
    }


def _multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label_names: List[str],
) -> Dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    y_pred = (y_prob >= threshold).astype(np.float32)
    hl = float(hamming_loss(y_true, y_pred))
    accuracy = float(1.0 - hl)  # Hamming accuracy (per-label correctness)
    # sample-wise exact-match accuracy
    exact = float((y_pred == y_true).all(axis=1).mean()) if len(y_true) else 0.0

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_label = {label_names[i]: float(f1_per[i]) for i in range(len(label_names))}
    precision_per_label = {label_names[i]: float(prec_per[i]) for i in range(len(label_names))}
    recall_per_label = {label_names[i]: float(rec_per[i]) for i in range(len(label_names))}

    # Metrics only on labels that appear in the batch (active)
    active_idx = [i for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0]
    if active_idx:
        subset_f1 = float(
            f1_score(y_true[:, active_idx], y_pred[:, active_idx], average="macro", zero_division=0)
        )
        subset_prec = float(
            precision_score(y_true[:, active_idx], y_pred[:, active_idx], average="macro", zero_division=0)
        )
        subset_rec = float(
            recall_score(y_true[:, active_idx], y_pred[:, active_idx], average="macro", zero_division=0)
        )
    else:
        subset_f1 = subset_prec = subset_rec = 0.0

    # Only compute AP on labels with at least one positive (avoids sklearn warnings / stalls)
    aps = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() <= 0:
            continue
        try:
            aps.append(float(average_precision_score(y_true[:, i], y_prob[:, i])))
        except ValueError:
            continue
    map_score = float(np.mean(aps)) if aps else 0.0

    return {
        "accuracy": accuracy,
        "exact_match_accuracy": exact,
        "hamming_loss": hl,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "subset_f1": subset_f1,
        "subset_precision": subset_prec,
        "subset_recall": subset_rec,
        "f1_per_label": f1_per_label,
        "precision_per_label": precision_per_label,
        "recall_per_label": recall_per_label,
        "map": map_score,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_classifier(
    data_dir: Union[str, Path],
    num_classes: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train the small-object classifier on ImageFolder-style data_dir.
    Returns metrics dict and saves best checkpoint + model_meta.
    """
    ensure_data_dirs()
    config = {
        "epochs": 30,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "image_size": 128,
        "val_split": 0.2,
        "pretrained": True,
        **(config or {}),
    }

    data_dir = Path(data_dir)
    append_training_log(f"Starting classifier training on {data_dir}")

    use_alb = True
    try:
        import albumentations  # noqa: F401
    except ImportError:
        use_alb = False

    if use_alb:
        full_ds = AlbumentationsImageFolder(data_dir, transform=None)
        classes = full_ds.classes
        if not classes:
            raise ValueError(f"No class folders found in {data_dir}")
        n = len(full_ds)
        if n < 2:
            raise ValueError("Need at least 2 images to train/validate.")

        n_val = max(1, int(n * config["val_split"]))
        n_train = n - n_val
        indices = list(range(n))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_tf = _classifier_transforms(True, config["image_size"])
        val_tf = _classifier_transforms(False, config["image_size"])

        class _Split(Dataset):
            def __init__(self, base, idxs, tf):
                self.base = base
                self.idxs = idxs
                self.tf = tf

            def __len__(self):
                return len(self.idxs)

            def __getitem__(self, i):
                path, label = self.base.samples[self.idxs[i]]
                image = np.array(Image.open(path).convert("RGB"))
                image = self.tf(image=image)["image"]
                return image, label

        train_ds = _Split(full_ds, train_idx, train_tf)
        val_ds = _Split(full_ds, val_idx, val_tf)
        num_classes = len(classes)
    else:
        # torchvision fallback
        raw = ImageFolder(data_dir)
        classes = list(raw.classes)
        num_classes = len(classes)
        n = len(raw)
        n_val = max(1, int(n * config["val_split"]))
        n_train = n - n_val
        train_raw, val_raw = random_split(
            raw, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        train_tf = _classifier_transforms(True, config["image_size"])
        val_tf = _classifier_transforms(False, config["image_size"])

        class _TVWrap(Dataset):
            def __init__(self, subset, tf):
                self.subset = subset
                self.tf = tf

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, i):
                path, label = self.subset.dataset.samples[self.subset.indices[i]]
                img = Image.open(path).convert("RGB")
                return self.tf(img), label

        train_ds = _TVWrap(train_raw, train_tf)
        val_ds = _TVWrap(val_raw, val_tf)

    device = get_device()
    model = build_small_object_classifier(num_classes, pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    best_acc = -1.0
    loss_curve: List[float] = []
    acc_curve: List[float] = []
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
        scheduler.step()
        epoch_loss = running_loss / max(n_seen, 1)
        loss_curve.append(epoch_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_targets: List[int] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())
        val_acc = correct / max(total, 1)
        acc_curve.append(val_acc)
        append_training_log(
            f"Classifier epoch {epoch}/{config['epochs']} loss={epoch_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Confusion matrix on final best model
    from sklearn.metrics import confusion_matrix

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(labels.tolist())
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes))).tolist()
    clf_metrics = metrics_from_confusion(cm, classes)

    torch.save(model.state_dict(), CLASSIFIER_WEIGHTS)
    meta = load_model_meta()
    meta["classes"] = classes
    meta["classifier"] = {
        "trained": True,
        "accuracy": best_acc,
        "best_val_accuracy": best_acc,
        "precision_macro": clf_metrics["precision_macro"],
        "recall_macro": clf_metrics["recall_macro"],
        "f1_macro": clf_metrics["f1_macro"],
        "per_class": clf_metrics["per_class"],
        "loss_curve": loss_curve,
        "accuracy_curve": acc_curve,
        "confusion_matrix": cm,
        "weights_file": CLASSIFIER_WEIGHTS.name,
        "weights_bytes": CLASSIFIER_WEIGHTS.stat().st_size if CLASSIFIER_WEIGHTS.exists() else None,
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in config.items() if k != "pretrained"},
        "architecture": "EfficientNet-B3 + CBAM + MLP head",
    }
    save_model_meta(meta)
    append_training_log(f"Classifier training complete. best_val_acc={best_acc:.4f}")

    return {
        "status": "completed",
        "accuracy": best_acc,
        "precision_macro": clf_metrics["precision_macro"],
        "recall_macro": clf_metrics["recall_macro"],
        "f1_macro": clf_metrics["f1_macro"],
        "per_class": clf_metrics["per_class"],
        "loss_curve": loss_curve,
        "accuracy_curve": acc_curve,
        "confusion_matrix": cm,
        "classes": classes,
        "weights_path": str(CLASSIFIER_WEIGHTS),
    }


def train_annotator(
    data_dir: Union[str, Path],
    labels: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train multi-label satellite annotator.
    Saves best checkpoint by validation mAP.
    """
    ensure_data_dirs()
    config = {
        "epochs": 30,
        "batch_size": 16,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "image_size": 224,
        "val_split": 0.2,
        "threshold": 0.5,
        "pretrained": True,
        **(config or {}),
    }

    meta = load_model_meta()
    label_names = list(labels or meta.get("labels") or DEFAULT_ENV_LABELS)
    data_dir = Path(data_dir)
    append_training_log(f"Starting annotator training on {data_dir}")

    train_tf = _annotator_transforms(True, config["image_size"])
    val_tf = _annotator_transforms(False, config["image_size"])

    full = SatelliteDataset(data_dir, label_names, transform=None, image_size=config["image_size"])
    if len(full) < 2:
        raise ValueError("Need at least 2 annotated satellite images to train/validate.")

    n = len(full)
    n_val = max(1, int(n * config["val_split"]))
    n_train = n - n_val
    indices = list(range(n))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    class _SatSplit(Dataset):
        def __init__(self, base: SatelliteDataset, idxs, tf):
            self.base = base
            self.idxs = idxs
            self.tf = tf

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            path, multi_hot = self.base.samples[self.idxs[i]]
            image = np.array(Image.open(path).convert("RGB"))
            if self.tf is not None:
                image = self.tf(image=image)["image"]
            else:
                image = cv2.resize(image, (config["image_size"], config["image_size"]))
                image = image.astype(np.float32) / 255.0
                image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            return image, torch.from_numpy(multi_hot)

    train_ds = _SatSplit(full, train_idx, train_tf)
    val_ds = _SatSplit(full, val_idx, val_tf)

    device = get_device()
    model = build_multilabel_annotator(len(label_names), pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Class-balanced positive weights for rare labels
    pos_counts = np.zeros(len(label_names), dtype=np.float64)
    for _, mh in (full.samples[i] for i in train_idx):
        pos_counts += mh
    n_train = max(len(train_idx), 1)
    neg_counts = n_train - pos_counts
    pos_w = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1.0), 1.0)
    pos_w = np.clip(pos_w, 0.5, 8.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    best_score = -1.0
    best_state = None
    loss_curve: List[float] = []
    map_curve: List[float] = []
    last_metrics: Dict[str, Any] = {}

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
        epoch_loss = running_loss / max(n_seen, 1)
        loss_curve.append(epoch_loss)
        scheduler.step()

        # Validation
        model.eval()
        all_true, all_prob = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_prob.append(probs)
                all_true.append(targets.numpy())
        y_true = np.concatenate(all_true, axis=0)
        y_prob = np.concatenate(all_prob, axis=0)
        metrics = _multilabel_metrics(y_true, y_prob, config["threshold"], label_names)
        map_curve.append(metrics["map"])
        last_metrics = metrics
        append_training_log(
            f"Annotator epoch {epoch}/{config['epochs']} loss={epoch_loss:.4f} "
            f"mAP={metrics['map']:.4f} acc={metrics['accuracy']:.4f} "
            f"subset_f1={metrics['subset_f1']:.4f}"
        )

        # Prefer high Hamming accuracy, break ties with mAP
        score = metrics["accuracy"] + 0.15 * metrics["map"]
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics with best weights
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(targets.numpy())
    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    final_metrics = _multilabel_metrics(y_true, y_prob, config["threshold"], label_names)

    torch.save(model.state_dict(), ANNOTATOR_WEIGHTS)
    meta = load_model_meta()
    meta["labels"] = label_names
    meta["threshold"] = config["threshold"]
    meta["annotator"] = {
        "trained": True,
        "accuracy": final_metrics["accuracy"],
        "exact_match_accuracy": final_metrics["exact_match_accuracy"],
        "map": final_metrics["map"],
        "hamming_loss": final_metrics["hamming_loss"],
        "f1_per_label": final_metrics["f1_per_label"],
        "precision_per_label": final_metrics["precision_per_label"],
        "recall_per_label": final_metrics["recall_per_label"],
        "f1_macro": final_metrics["f1_macro"],
        "precision_macro": final_metrics["precision_macro"],
        "recall_macro": final_metrics["recall_macro"],
        "subset_f1": final_metrics["subset_f1"],
        "subset_precision": final_metrics["subset_precision"],
        "subset_recall": final_metrics["subset_recall"],
        "loss_curve": loss_curve,
        "map_curve": map_curve,
        "weights_file": ANNOTATOR_WEIGHTS.name,
        "weights_bytes": ANNOTATOR_WEIGHTS.stat().st_size if ANNOTATOR_WEIGHTS.exists() else None,
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in config.items() if k != "pretrained"},
        "architecture": "EfficientNet-B3 + CBAM + MLP multi-label head",
    }
    save_model_meta(meta)
    append_training_log(
        f"Annotator training complete. accuracy={final_metrics['accuracy']:.4f} "
        f"mAP={final_metrics['map']:.4f} subset_f1={final_metrics['subset_f1']:.4f}"
    )

    return {
        "status": "completed",
        "accuracy": final_metrics["accuracy"],
        "map": final_metrics["map"],
        "hamming_loss": final_metrics["hamming_loss"],
        "subset_f1": final_metrics["subset_f1"],
        "f1_per_label": final_metrics["f1_per_label"],
        "loss_curve": loss_curve,
        "map_curve": map_curve,
        "labels": label_names,
        "weights_path": str(ANNOTATOR_WEIGHTS),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_class(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Softmax classification; returns {class, confidence, top3, probabilities}."""
    device = device or get_device()
    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    logits = model(image_tensor)
    probs = F.softmax(logits, dim=1)[0]
    conf, pred_idx = probs.max(dim=0)
    topk = min(3, probs.numel())
    top_vals, top_idxs = probs.topk(topk)

    if class_names is None:
        meta = load_model_meta()
        class_names = meta.get("classes") or [f"class_{i}" for i in range(probs.numel())]

    top3 = [
        {"class": class_names[i] if i < len(class_names) else f"class_{i}", "confidence": float(v)}
        for v, i in zip(top_vals.tolist(), top_idxs.tolist())
    ]
    pred_name = class_names[int(pred_idx)] if int(pred_idx) < len(class_names) else f"class_{int(pred_idx)}"

    return {
        "class": pred_name,
        "confidence": float(conf),
        "top3": top3,
        "probabilities": {
            (class_names[i] if i < len(class_names) else f"class_{i}"): float(probs[i])
            for i in range(probs.numel())
        },
    }


@torch.no_grad()
def predict_labels(
    model: nn.Module,
    image_tensor: torch.Tensor,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Multi-label sigmoid scores; returns labels with scores and above_threshold flags."""
    device = device or get_device()
    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    logits = model(image_tensor)
    scores = torch.sigmoid(logits)[0].cpu().tolist()

    if label_names is None:
        meta = load_model_meta()
        label_names = meta.get("labels") or DEFAULT_ENV_LABELS

    labels = []
    for i, name in enumerate(label_names):
        score = float(scores[i]) if i < len(scores) else 0.0
        labels.append(
            {
                "name": name,
                "score": score,
                "above_threshold": score >= threshold,
            }
        )

    return {"labels": labels, "threshold": threshold}


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


def _overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB uint8 image; return RGB uint8."""
    h, w = image_rgb.shape[:2]
    heat = cv2.resize(heatmap.astype(np.float32), (w, h))
    heat = np.clip(heat, 0, 1)
    heat_u8 = np.uint8(255 * heat)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * color + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


def _image_to_base64_png(image_rgb: np.ndarray) -> str:
    pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    class_idx: Optional[int] = None,
    original_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute GradCAM heatmap (H x W float in [0,1]) via a manual GradCAM
    implementation (torchcam optional; hooks always cleaned up).
    """
    device = get_device()
    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    layer = target_layer
    if layer is None:
        layer = getattr(model, "target_layer", None)
    if layer is None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                layer = m
    if layer is None:
        raise RuntimeError("No target layer found for GradCAM")

    heatmap = _manual_gradcam(model, image_tensor, layer, class_idx)

    heatmap = np.nan_to_num(heatmap.astype(np.float32))
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def _manual_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    layer: nn.Module,
    class_idx: Optional[int],
) -> np.ndarray:
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, o):
        activations["value"] = o

    def bwd_hook(_m, _gi, go):
        gradients["value"] = go[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        # Clone so we don't permanently alter caller's tensor flags
        x = image_tensor.detach().clone().requires_grad_(True)
        out = model(x)
        if class_idx is None:
            if isinstance(model, MultiLabelAnnotator):
                class_idx = int(torch.sigmoid(out)[0].argmax().item())
            else:
                class_idx = int(out.argmax(dim=1).item())

        score = out[0, class_idx]
        score.backward()

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_np = cam.squeeze().detach().cpu().numpy()
        if was_training:
            model.train()
        return cam_np
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)

def get_gradcam_overlay_base64(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    target_layer: Optional[nn.Module] = None,
    class_idx: Optional[int] = None,
) -> str:
    """Return base64 PNG of GradCAM overlay on the original RGB image."""
    heatmap = get_gradcam(model, image_tensor, target_layer=target_layer, class_idx=class_idx)
    overlay = _overlay_heatmap_on_image(original_rgb, heatmap)
    return _image_to_base64_png(overlay)


# ---------------------------------------------------------------------------
# Sliding window patches
# ---------------------------------------------------------------------------


def sliding_window_patches(
    image: Union[np.ndarray, Image.Image, str, Path],
    patch_size: int = 128,
    stride: int = 64,
) -> List[Dict[str, Any]]:
    """
    Extract sliding-window patches from a full image.
    Returns list of {patch: np.ndarray RGB, bbox: [x1,y1,x2,y2], patch_id: int}.
    """
    if isinstance(image, (str, Path)):
        rgb = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 3:
            # assume already RGB if passed as ndarray from PIL path; callers should pass RGB
            pass

    h, w = rgb.shape[:2]
    patches: List[Dict[str, Any]] = []
    patch_id = 0

    # Ensure at least one patch even for small images
    ys = list(range(0, max(h - patch_size + 1, 1), stride))
    xs = list(range(0, max(w - patch_size + 1, 1), stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    # Always include bottom-right aligned crop if image larger than patch
    if h >= patch_size and (h - patch_size) not in ys:
        ys.append(h - patch_size)
    if w >= patch_size and (w - patch_size) not in xs:
        xs.append(w - patch_size)

    for y in ys:
        for x in xs:
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size) if h >= patch_size else 0
            x1 = max(0, x2 - patch_size) if w >= patch_size else 0
            # For tiny images, pad
            crop = rgb[y1:y2, x1:x2]
            if crop.shape[0] < patch_size or crop.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=rgb.dtype)
                padded[: crop.shape[0], : crop.shape[1]] = crop
                crop = padded
                x2 = x1 + patch_size
                y2 = y1 + patch_size
            patches.append(
                {
                    "patch_id": patch_id,
                    "patch": crop,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
            patch_id += 1

    return patches


# ---------------------------------------------------------------------------
# Model load helpers (used by API later)
# ---------------------------------------------------------------------------


def load_classifier(
    num_classes: Optional[int] = None,
    weights_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[nn.Module], List[str]]:
    ensure_data_dirs()
    meta = load_model_meta()
    classes = meta.get("classes") or []
    n = num_classes or len(classes)
    weights_path = Path(weights_path) if weights_path else CLASSIFIER_WEIGHTS
    if n == 0 or not weights_path.exists():
        return None, classes
    device = device or get_device()
    model = build_small_object_classifier(n, pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes


def load_annotator(
    num_labels: Optional[int] = None,
    weights_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[nn.Module], List[str]]:
    ensure_data_dirs()
    meta = load_model_meta()
    labels = meta.get("labels") or list(DEFAULT_ENV_LABELS)
    n = num_labels or len(labels)
    weights_path = Path(weights_path) if weights_path else ANNOTATOR_WEIGHTS
    if not weights_path.exists():
        return None, labels
    device = device or get_device()
    model = build_multilabel_annotator(n, pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, labels


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Build models and run forward passes with dummy tensors (no pretrained download if offline)."""
    print("=" * 60)
    print("model_service smoke test")
    print("=" * 60)
    device = get_device()
    print(f"device: {device}")

    # --- CBAM ---
    print("\n[1] CBAM block")
    cbam = CBAM(channels=64)
    x = torch.randn(2, 64, 16, 16)
    y = cbam(x)
    assert y.shape == x.shape, y.shape
    print(f"  CBAM OK: {tuple(x.shape)} → {tuple(y.shape)}")

    # --- Classifier (no pretrained weights for fast offline test) ---
    print("\n[2] SmallObjectClassifier")
    clf = build_small_object_classifier(num_classes=4, pretrained=False).to(device)
    inp = torch.randn(2, 3, 128, 128, device=device)
    logits = clf(inp)
    assert logits.shape == (2, 4), logits.shape
    print(f"  Classifier OK: input {tuple(inp.shape)} → logits {tuple(logits.shape)}")
    pred = predict_class(clf, inp[:1], class_names=["vehicle", "ship", "aircraft", "building"])
    print(f"  predict_class: {pred['class']} ({pred['confidence']:.3f}) top3={pred['top3']}")

    # --- Annotator ---
    print("\n[3] MultiLabelAnnotator")
    n_labels = len(DEFAULT_ENV_LABELS)
    ann = build_multilabel_annotator(num_labels=n_labels, pretrained=False).to(device)
    inp2 = torch.randn(2, 3, 224, 224, device=device)
    logits2 = ann(inp2)
    assert logits2.shape == (2, n_labels), logits2.shape
    print(f"  Annotator OK: input {tuple(inp2.shape)} → logits {tuple(logits2.shape)}")
    lab = predict_labels(ann, inp2[:1], threshold=0.5, label_names=DEFAULT_ENV_LABELS)
    above = [l["name"] for l in lab["labels"] if l["above_threshold"]]
    print(f"  predict_labels: {len(lab['labels'])} labels, above_threshold={above[:5]}...")

    # --- Preprocess ---
    print("\n[4] detect_and_preprocess")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (200, 180, 3), dtype=np.uint8))
    t = detect_and_preprocess(dummy_img, target_size=(128, 128))
    assert t.shape == (1, 3, 128, 128), t.shape
    print(f"  preprocess OK: {tuple(t.shape)}")

    # --- Sliding window ---
    print("\n[5] sliding_window_patches")
    big = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    patches = sliding_window_patches(big, patch_size=128, stride=64)
    assert len(patches) > 0
    assert patches[0]["patch"].shape == (128, 128, 3)
    print(f"  sliding_window OK: {len(patches)} patches, bbox0={patches[0]['bbox']}")

    # --- GradCAM ---
    print("\n[6] get_gradcam")
    heat = get_gradcam(clf, t.to(device), target_layer=clf.target_layer)
    assert heat.ndim == 2
    print(f"  GradCAM OK: heatmap shape {heat.shape}, range [{heat.min():.3f}, {heat.max():.3f}]")
    overlay_b64 = get_gradcam_overlay_base64(
        clf, t.to(device), np.array(dummy_img.resize((128, 128))), target_layer=clf.target_layer
    )
    assert len(overlay_b64) > 100
    print(f"  GradCAM overlay base64 length={len(overlay_b64)}")

    # --- Data dirs ---
    print("\n[7] ensure_data_dirs")
    ensure_data_dirs()
    assert MODEL_META_PATH.exists()
    print(f"  data dirs OK, meta at {MODEL_META_PATH}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
