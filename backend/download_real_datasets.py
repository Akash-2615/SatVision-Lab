"""
Download real remote-sensing datasets and prepare train/test splits.

Task 1 — NWPU VHR-10 (Google Earth VHR aerial imagery)
  Crop airplane / ship / vehicle / storage-tank(→building) patches.

Task 2 — EuroSAT RGB (Sentinel-2 64×64 land-cover scenes)
  Map single land-cover classes → multi-label environmental tags.

Outputs:
  backend/data/datasets/small_objects/<class>/*.jpg   (train)
  backend/data/datasets/satellite/{images,annotations}/
  backend/data/test/small_objects/<class>/*.jpg       (held-out)
  backend/data/test/satellite/{images,annotations}/
"""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
TRAIN_SO = DATA / "datasets" / "small_objects"
TRAIN_SAT_IMG = DATA / "datasets" / "satellite" / "images"
TRAIN_SAT_ANN = DATA / "datasets" / "satellite" / "annotations"
TEST_SO = DATA / "test" / "small_objects"
TEST_SAT_IMG = DATA / "test" / "satellite" / "images"
TEST_SAT_ANN = DATA / "test" / "satellite" / "annotations"

NWPU_URL = "https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip"
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
# Fallback mirror (DFKI / original hosting often flaky)
EUROSAT_FALLBACK = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip"

# NWPU class ids (1-indexed in many GT files): airplane, ship, storage tank, ..., vehicle
# We map: airplane→aircraft, ship→ship, vehicle→vehicle, storage tank→building
NWPU_ID_TO_CLASS = {
    1: "aircraft",
    2: "ship",
    3: "building",  # storage tank as built structure proxy
    10: "vehicle",
}

# EuroSAT → multi-label environmental tags used by the annotator
EUROSAT_TO_LABELS: Dict[str, List[str]] = {
    "AnnualCrop": ["agriculture"],
    "Forest": ["dense_forest"],
    "HerbaceousVegetation": ["sparse_vegetation"],
    "Highway": ["urban_area", "barren_land"],
    "Industrial": ["urban_area", "barren_land"],
    "Pasture": ["sparse_vegetation", "agriculture"],
    "PermanentCrop": ["agriculture"],
    "Residential": ["urban_area"],
    "River": ["water_body", "wetland"],
    "SeaLake": ["water_body"],
}

# Subset sizes (keep CPU training tractable)
MAX_PER_SO_CLASS_TRAIN = 80
MAX_PER_SO_CLASS_TEST = 20
MAX_SAT_PER_CLASS_TRAIN = 40
MAX_SAT_PER_CLASS_TEST = 10

RNG = random.Random(42)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  already have {dest.name}")
        return dest
    print(f"  downloading {url}")
    print(f"  → {dest}")
    urlretrieve(url, dest)
    return dest


def unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"  already extracted → {out_dir}")
        return out_dir
    print(f"  extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text("ok")
    return out_dir


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_nwpu_gt(gt_path: Path) -> List[Tuple[int, int, int, int, int]]:
    """Return list of (x1,y1,x2,y2,class_id)."""
    boxes = []
    text = gt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        # formats: (x1,y1),(x2,y2),class  OR  x1,y1,x2,y2,class
        line = line.replace("(", "").replace(")", "")
        parts = [p.strip() for p in line.replace(" ", "").split(",") if p.strip()]
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2, cid = map(int, parts[:5])
        except ValueError:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2, cid))
    return boxes


def find_nwpu_roots(extracted: Path) -> Tuple[Path, Path]:
    """Locate positive image folder and ground-truth folder."""
    img_dir = None
    gt_dir = None
    for p in extracted.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name.lower().replace(" ", "")
        if "positive" in name and "image" in name:
            img_dir = p
        if name in {"groundtruth", "ground_truth", "gt"} or "groundtruth" in name:
            gt_dir = p
    if img_dir is None or gt_dir is None:
        # fallback: list structure
        dirs = [str(p.relative_to(extracted)) for p in extracted.rglob("*") if p.is_dir()]
        raise FileNotFoundError(f"Could not find NWPU image/gt folders. Found dirs: {dirs[:40]}")
    return img_dir, gt_dir


def prepare_nwpu(extracted: Path) -> Dict[str, List[Path]]:
    img_dir, gt_dir = find_nwpu_roots(extracted)
    print(f"  NWPU images: {img_dir}")
    print(f"  NWPU gt:     {gt_dir}")

    crops: Dict[str, List[Image.Image]] = defaultdict(list)
    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    for img_path in img_files:
        stem = img_path.stem
        # GT often named 001.txt matching 001.jpg
        candidates = [
            gt_dir / f"{stem}.txt",
            gt_dir / f"{stem}.xml",
            gt_dir / f"{int(stem):03d}.txt" if stem.isdigit() else gt_dir / f"{stem}.txt",
        ]
        gt_path = next((c for c in candidates if c.exists()), None)
        if gt_path is None:
            # try any file starting with stem
            matches = list(gt_dir.glob(f"{stem}*"))
            gt_path = matches[0] if matches else None
        if gt_path is None or gt_path.suffix.lower() == ".xml":
            # skip XML for simplicity; most NWPU releases use .txt
            if gt_path is None:
                continue
        boxes = parse_nwpu_gt(gt_path)
        if not boxes:
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        for x1, y1, x2, y2, cid in boxes:
            cls = NWPU_ID_TO_CLASS.get(cid)
            if cls is None:
                continue
            # pad small boxes a bit
            bw, bh = x2 - x1, y2 - y1
            pad = max(4, int(0.15 * max(bw, bh)))
            xa = max(0, x1 - pad)
            ya = max(0, y1 - pad)
            xb = min(W, x2 + pad)
            yb = min(H, y2 + pad)
            if xb - xa < 16 or yb - ya < 16:
                continue
            crop = img.crop((xa, ya, xb, yb)).resize((128, 128), Image.BILINEAR)
            crops[cls].append(crop)

    # Save all crops temporarily then split
    tmp = RAW / "nwpu_crops"
    if tmp.exists():
        shutil.rmtree(tmp)
    by_class: Dict[str, List[Path]] = {}
    for cls, imgs in crops.items():
        d = tmp / cls
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, im in enumerate(imgs):
            p = d / f"nwpu_{cls}_{i:04d}.jpg"
            im.save(p, quality=92)
            paths.append(p)
        by_class[cls] = paths
        print(f"  cropped {cls}: {len(paths)}")
    return by_class


def split_and_write_so(by_class: Dict[str, List[Path]]) -> None:
    clear_dir(TRAIN_SO)
    clear_dir(TEST_SO)
    for cls, paths in by_class.items():
        paths = list(paths)
        RNG.shuffle(paths)
        n_test = min(MAX_PER_SO_CLASS_TEST, max(1, int(0.2 * len(paths))))
        n_train = min(MAX_PER_SO_CLASS_TRAIN, len(paths) - n_test)
        if n_train < 2:
            n_train = max(0, len(paths) - n_test)
        test_paths = paths[:n_test]
        train_paths = paths[n_test : n_test + n_train]
        (TRAIN_SO / cls).mkdir(parents=True, exist_ok=True)
        (TEST_SO / cls).mkdir(parents=True, exist_ok=True)
        for p in train_paths:
            shutil.copy2(p, TRAIN_SO / cls / p.name)
        for p in test_paths:
            shutil.copy2(p, TEST_SO / cls / p.name)
        print(f"  {cls}: train={len(train_paths)} test={len(test_paths)}")


def find_eurosat_root(extracted: Path) -> Path:
    # Expect class subfolders
    for p in extracted.rglob("Forest"):
        if p.is_dir() and (p.parent / "SeaLake").exists():
            return p.parent
    for p in extracted.rglob("AnnualCrop"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError("EuroSAT class folders not found")


def prepare_eurosat(extracted: Path) -> None:
    root = find_eurosat_root(extracted)
    print(f"  EuroSAT root: {root}")
    clear_dir(TRAIN_SAT_IMG)
    clear_dir(TRAIN_SAT_ANN)
    clear_dir(TEST_SAT_IMG)
    clear_dir(TEST_SAT_ANN)

    # ensure empty dirs exist
    for d in (TRAIN_SAT_IMG, TRAIN_SAT_ANN, TEST_SAT_IMG, TEST_SAT_ANN):
        d.mkdir(parents=True, exist_ok=True)

    for cls_name, labels in EUROSAT_TO_LABELS.items():
        folder = root / cls_name
        if not folder.exists():
            print(f"  skip missing class {cls_name}")
            continue
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        RNG.shuffle(files)
        n_test = min(MAX_SAT_PER_CLASS_TEST, max(1, int(0.2 * min(len(files), MAX_SAT_PER_CLASS_TRAIN + MAX_SAT_PER_CLASS_TEST))))
        take = min(len(files), MAX_SAT_PER_CLASS_TRAIN + MAX_SAT_PER_CLASS_TEST)
        files = files[:take]
        test_files = files[:n_test]
        train_files = files[n_test:]

        for split_files, img_dir, ann_dir, prefix in [
            (train_files, TRAIN_SAT_IMG, TRAIN_SAT_ANN, "train"),
            (test_files, TEST_SAT_IMG, TEST_SAT_ANN, "test"),
        ]:
            for i, src in enumerate(split_files):
                # upscale slightly for annotator 224 input quality
                im = Image.open(src).convert("RGB").resize((128, 128), Image.BILINEAR)
                name = f"eurosat_{cls_name}_{prefix}_{i:03d}.jpg"
                im.save(img_dir / name, quality=92)
                with open(ann_dir / f"{Path(name).stem}.json", "w") as f:
                    json.dump({"labels": labels, "source_class": cls_name}, f, indent=2)

        print(
            f"  {cls_name}: train={len(train_files)} test={len(test_files)} → {labels}"
        )


def download_eurosat() -> Path:
    zip_path = RAW / "EuroSAT_RGB.zip"
    try:
        download(EUROSAT_URL, zip_path)
    except Exception as e1:
        print(f"  Zenodo failed ({e1}); trying fallback…")
        try:
            download(EUROSAT_FALLBACK, zip_path)
        except Exception as e2:
            # torchvision cache / HF dataset
            print(f"  Fallback zip failed ({e2}); using torchvision EuroSAT…")
            return download_eurosat_torchvision()
    out = RAW / "eurosat"
    unzip(zip_path, out)
    return out


def download_eurosat_torchvision() -> Path:
    """Pull EuroSAT via torchvision into RAW/eurosat_tv."""
    import torchvision

    out = RAW / "eurosat_tv"
    out.mkdir(parents=True, exist_ok=True)
    # torchvision downloads into root and extracts
    ds = torchvision.datasets.EuroSAT(root=str(RAW / "eurosat_tv_download"), download=True)
    # Reorganize into class folders if needed
    # EuroSAT structure after download: <root>/eurosat/2750/<Class>/*.jpg
    extracted = RAW / "eurosat_tv_download"
    # find class root
    root = find_eurosat_root(extracted)
    # copy tree to out for consistency
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(root, out / "2750")
    print(f"  torchvision EuroSAT ready at {out}")
    _ = ds  # silence unused
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1) NWPU VHR-10 — small object crops")
    print("=" * 60)
    nwpu_zip = download(NWPU_URL, RAW / "NWPU-VHR-10.zip")
    nwpu_dir = unzip(nwpu_zip, RAW / "nwpu")
    by_class = prepare_nwpu(nwpu_dir)
    required = {"aircraft", "ship", "vehicle", "building"}
    missing = required - set(by_class.keys())
    if missing:
        raise RuntimeError(f"Missing NWPU classes after crop: {missing}. Got: {list(by_class)}")
    split_and_write_so(by_class)

    print("=" * 60)
    print("2) EuroSAT RGB — satellite multi-label scenes")
    print("=" * 60)
    eurosat_dir = download_eurosat()
    prepare_eurosat(eurosat_dir)

    # Write split manifest
    manifest = {
        "small_objects": {
            cls: {
                "train": len(list((TRAIN_SO / cls).glob("*.jpg"))),
                "test": len(list((TEST_SO / cls).glob("*.jpg"))),
            }
            for cls in sorted(p.name for p in TRAIN_SO.iterdir() if p.is_dir())
        },
        "satellite": {
            "train": len(list(TRAIN_SAT_IMG.glob("*.jpg"))),
            "test": len(list(TEST_SAT_IMG.glob("*.jpg"))),
        },
        "sources": {
            "small_objects": "NWPU VHR-10 (https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip)",
            "satellite": "EuroSAT RGB (Sentinel-2)",
        },
    }
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("=" * 60)
    print("DONE")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
