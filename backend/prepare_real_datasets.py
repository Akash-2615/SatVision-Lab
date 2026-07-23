"""
Prepare real remote-sensing datasets already downloaded under data/raw/:

  Task 1: NWPU VHR-10 object crops (airplane/ship/vehicle/storage_tank)
  Task 2: EuroSAT RGB Sentinel-2 scenes → multi-label environmental tags

Creates train (data/datasets/) and held-out test (data/test/) splits.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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

NWPU_ZIP = RAW / "NWPU-VHR-10.zip"
NWPU_ANN = RAW / "vhr10_annotations.json"
NWPU_EXTRACT = RAW / "nwpu"
EUROSAT_ROOT = RAW / "eurosat_tv_download" / "eurosat" / "2750"

# COCO category id → our class names
COCO_TO_CLASS = {
    1: "aircraft",      # airplane
    2: "ship",
    3: "building",      # storage_tank as built structure
    10: "vehicle",
}

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

MAX_PER_SO_CLASS_TRAIN = 180
MAX_PER_SO_CLASS_TEST = 40
MAX_SAT_PER_CLASS_TRAIN = 120
MAX_SAT_PER_CLASS_TEST = 24

RNG = random.Random(42)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_nwpu_extracted() -> Path:
    marker = NWPU_EXTRACT / ".extracted"
    if not marker.exists():
        print("Extracting NWPU VHR-10…")
        NWPU_EXTRACT.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(NWPU_ZIP, "r") as zf:
            zf.extractall(NWPU_EXTRACT)
        marker.write_text("ok")
    img_dir = NWPU_EXTRACT / "NWPU VHR-10 dataset" / "positive image set"
    if not img_dir.exists():
        # search
        matches = list(NWPU_EXTRACT.rglob("positive image set"))
        if not matches:
            raise FileNotFoundError("positive image set not found")
        img_dir = matches[0]
    return img_dir


def prepare_nwpu_crops(img_dir: Path) -> Dict[str, List[Path]]:
    coco = json.loads(NWPU_ANN.read_text())
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    # group anns by image
    anns_by_img: Dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    tmp = RAW / "nwpu_crops"
    if tmp.exists():
        shutil.rmtree(tmp)
    counts = defaultdict(int)
    paths_by_class: Dict[str, List[Path]] = defaultdict(list)

    for image_id, anns in anns_by_img.items():
        fname = id_to_file[image_id]
        # file_name may be like "001.jpg" or with path
        fname = Path(fname).name
        img_path = img_dir / fname
        if not img_path.exists():
            # try zero-padded alternatives
            stem = Path(fname).stem
            if stem.isdigit():
                alt = img_dir / f"{int(stem)}.jpg"
                if alt.exists():
                    img_path = alt
                else:
                    alt = img_dir / f"{int(stem):03d}.jpg"
                    if alt.exists():
                        img_path = alt
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        for ann in anns:
            cid = ann["category_id"]
            cls = COCO_TO_CLASS.get(cid)
            if cls is None:
                continue
            # COCO bbox: [x, y, w, h]
            x, y, w, h = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            pad = max(4, int(0.15 * max(w, h)))
            xa, ya = max(0, x1 - pad), max(0, y1 - pad)
            xb, yb = min(W, x2 + pad), min(H, y2 + pad)
            if xb - xa < 16 or yb - ya < 16:
                continue
            crop = img.crop((xa, ya, xb, yb)).resize((128, 128), Image.BILINEAR)
            out_dir = tmp / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = counts[cls]
            out = out_dir / f"nwpu_{cls}_{idx:04d}.jpg"
            crop.save(out, quality=92)
            paths_by_class[cls].append(out)
            counts[cls] += 1

    for cls, n in sorted(counts.items()):
        print(f"  cropped {cls}: {n}")
    return dict(paths_by_class)


def split_so(by_class: Dict[str, List[Path]]) -> None:
    clear_dir(TRAIN_SO)
    clear_dir(TEST_SO)
    for cls, paths in by_class.items():
        paths = list(paths)
        RNG.shuffle(paths)
        n_test = min(MAX_PER_SO_CLASS_TEST, max(1, int(0.2 * len(paths))))
        n_train = min(MAX_PER_SO_CLASS_TRAIN, len(paths) - n_test)
        test_paths = paths[:n_test]
        train_paths = paths[n_test : n_test + n_train]
        (TRAIN_SO / cls).mkdir(parents=True, exist_ok=True)
        (TEST_SO / cls).mkdir(parents=True, exist_ok=True)
        for p in train_paths:
            shutil.copy2(p, TRAIN_SO / cls / p.name)
        for p in test_paths:
            shutil.copy2(p, TEST_SO / cls / p.name)
        print(f"  {cls}: train={len(train_paths)} test={len(test_paths)}")


def prepare_eurosat() -> None:
    if not EUROSAT_ROOT.exists():
        raise FileNotFoundError(f"EuroSAT not found at {EUROSAT_ROOT}")
    clear_dir(TRAIN_SAT_IMG)
    clear_dir(TRAIN_SAT_ANN)
    clear_dir(TEST_SAT_IMG)
    clear_dir(TEST_SAT_ANN)
    for d in (TRAIN_SAT_IMG, TRAIN_SAT_ANN, TEST_SAT_IMG, TEST_SAT_ANN):
        d.mkdir(parents=True, exist_ok=True)

    for cls_name, labels in EUROSAT_TO_LABELS.items():
        folder = EUROSAT_ROOT / cls_name
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        RNG.shuffle(files)
        take = min(len(files), MAX_SAT_PER_CLASS_TRAIN + MAX_SAT_PER_CLASS_TEST)
        files = files[:take]
        n_test = min(MAX_SAT_PER_CLASS_TEST, max(1, int(0.2 * take)))
        test_files = files[:n_test]
        train_files = files[n_test:]

        for split_files, img_dir, ann_dir, prefix in [
            (train_files, TRAIN_SAT_IMG, TRAIN_SAT_ANN, "train"),
            (test_files, TEST_SAT_IMG, TEST_SAT_ANN, "test"),
        ]:
            for i, src in enumerate(split_files):
                im = Image.open(src).convert("RGB").resize((128, 128), Image.BILINEAR)
                name = f"eurosat_{cls_name}_{prefix}_{i:03d}.jpg"
                im.save(img_dir / name, quality=92)
                with open(ann_dir / f"{Path(name).stem}.json", "w") as f:
                    json.dump({"labels": labels, "source_class": cls_name}, f, indent=2)
        print(f"  {cls_name}: train={len(train_files)} test={len(test_files)} → {labels}")


def main() -> None:
    assert NWPU_ZIP.exists(), f"Missing {NWPU_ZIP}"
    assert NWPU_ANN.exists(), f"Missing {NWPU_ANN}"

    print("=" * 60)
    print("NWPU VHR-10 → small-object crops")
    print("=" * 60)
    img_dir = ensure_nwpu_extracted()
    print(f"  images: {img_dir}")
    by_class = prepare_nwpu_crops(img_dir)
    required = {"aircraft", "ship", "vehicle", "building"}
    missing = required - set(by_class)
    if missing:
        raise RuntimeError(f"Missing classes {missing}, got {list(by_class)}")
    split_so(by_class)

    print("=" * 60)
    print("EuroSAT RGB → satellite multi-label")
    print("=" * 60)
    prepare_eurosat()

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
            "small_objects": "NWPU VHR-10 (torchgeo HF mirror + COCO annotations)",
            "satellite": "EuroSAT RGB via torchvision (Sentinel-2)",
        },
    }
    with open(DATA / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("=" * 60)
    print(json.dumps(manifest, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
