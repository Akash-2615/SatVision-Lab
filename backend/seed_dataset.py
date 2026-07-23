"""
Generate a small synthetic training dataset for the lab exercise.
Creates distinguishable color/texture patterns so models can learn something
even without real satellite imagery.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SMALL = ROOT / "data" / "datasets" / "small_objects"
SAT_IMG = ROOT / "data" / "datasets" / "satellite" / "images"
SAT_ANN = ROOT / "data" / "datasets" / "satellite" / "annotations"

CLASSES = {
    "vehicle": (220, 60, 60),       # red-ish blob
    "ship": (40, 100, 220),         # blue elongated
    "aircraft": (240, 220, 60),     # yellow cross
    "building": (120, 120, 130),    # gray rectangle grid
}

# Satellite scenes: background color + label set
SCENES = [
    {"name": "coastal_urban", "bg": (30, 90, 160), "labels": ["water_body", "urban_area", "cloud_cover"]},
    {"name": "forest_agri", "bg": (34, 120, 40), "labels": ["dense_forest", "agriculture", "sparse_vegetation"]},
    {"name": "desert_barren", "bg": (210, 180, 100), "labels": ["desert", "barren_land"]},
    {"name": "flood_wetland", "bg": (50, 100, 140), "labels": ["flood", "wetland", "water_body"]},
    {"name": "wildfire_smoke", "bg": (80, 50, 30), "labels": ["wildfire", "barren_land", "cloud_cover"]},
    {"name": "snow_mountain", "bg": (230, 235, 245), "labels": ["snow_ice", "barren_land"]},
    {"name": "farmland", "bg": (140, 170, 60), "labels": ["agriculture", "sparse_vegetation", "urban_area"]},
    {"name": "dense_city", "bg": (90, 90, 95), "labels": ["urban_area", "barren_land"]},
    {"name": "wetland_marsh", "bg": (60, 130, 100), "labels": ["wetland", "sparse_vegetation", "water_body"]},
    {"name": "cloudy_ocean", "bg": (70, 110, 170), "labels": ["water_body", "cloud_cover"]},
]


def _noise(img: Image.Image, amount: float = 18.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr += np.random.randn(*arr.shape) * amount
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_small_object(class_name: str, size: int = 128) -> Image.Image:
    color = CLASSES[class_name]
    bg = (
        random.randint(20, 80),
        random.randint(40, 100),
        random.randint(20, 80),
    )
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2 + random.randint(-10, 10), size // 2 + random.randint(-10, 10)
    s = random.randint(18, 40)

    if class_name == "vehicle":
        draw.rectangle([cx - s, cy - s // 2, cx + s, cy + s // 2], fill=color)
        draw.ellipse([cx - s // 2, cy + s // 3, cx - s // 4, cy + s // 2 + 4], fill=(20, 20, 20))
        draw.ellipse([cx + s // 4, cy + s // 3, cx + s // 2, cy + s // 2 + 4], fill=(20, 20, 20))
    elif class_name == "ship":
        draw.polygon(
            [(cx - s * 1.5, cy), (cx + s * 1.5, cy - s // 3), (cx + s * 1.5, cy + s // 3)],
            fill=color,
        )
        draw.rectangle([cx - 4, cy - s, cx + 4, cy], fill=(200, 200, 200))
    elif class_name == "aircraft":
        draw.line([(cx - s, cy), (cx + s, cy)], fill=color, width=6)
        draw.line([(cx, cy - s // 2), (cx, cy + s)], fill=color, width=5)
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(250, 250, 250))
    else:  # building
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=color)
        for i in range(-s + 4, s - 2, 10):
            for j in range(-s + 4, s - 2, 10):
                draw.rectangle([cx + i, cy + j, cx + i + 5, cy + j + 5], fill=(200, 200, 80))

    return _noise(img, 12)


def make_satellite(scene: dict, size: int = 256) -> Image.Image:
    bg = scene["bg"]
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    labels = set(scene["labels"])

    if "urban_area" in labels:
        for _ in range(25):
            x, y = random.randint(0, size - 20), random.randint(0, size - 20)
            w, h = random.randint(8, 22), random.randint(8, 22)
            draw.rectangle([x, y, x + w, y + h], fill=(110, 110, 120))

    if "dense_forest" in labels or "sparse_vegetation" in labels:
        n = 80 if "dense_forest" in labels else 30
        for _ in range(n):
            x, y = random.randint(0, size - 8), random.randint(0, size - 8)
            draw.ellipse([x, y, x + 8, y + 8], fill=(20, 100 + random.randint(0, 40), 30))

    if "agriculture" in labels:
        for i in range(0, size, 20):
            draw.line([(0, i), (size, i)], fill=(150, 180, 50), width=2)

    if "water_body" in labels or "flood" in labels or "wetland" in labels:
        for _ in range(5):
            x, y = random.randint(0, size // 2), random.randint(0, size // 2)
            draw.ellipse(
                [x, y, x + random.randint(40, 120), y + random.randint(30, 90)],
                fill=(40, 90, 180),
            )

    if "desert" in labels or "barren_land" in labels:
        for _ in range(40):
            x, y = random.randint(0, size - 1), random.randint(0, size - 1)
            draw.point((x, y), fill=(230, 200, 120))

    if "cloud_cover" in labels:
        for _ in range(8):
            x, y = random.randint(0, size - 40), random.randint(0, size // 2)
            draw.ellipse([x, y, x + 50, y + 25], fill=(220, 220, 230))

    if "wildfire" in labels:
        for _ in range(15):
            x, y = random.randint(0, size - 15), random.randint(size // 3, size - 15)
            draw.ellipse([x, y, x + 12, y + 12], fill=(255, 80, 20))

    if "snow_ice" in labels:
        for _ in range(60):
            x, y = random.randint(0, size - 1), random.randint(0, size - 1)
            draw.point((x, y), fill=(255, 255, 255))

    return _noise(img, 10)


def main(n_per_class: int = 24, n_variants: int = 2) -> None:
    random.seed(42)
    np.random.seed(42)

    print("Generating small-object patches...")
    for cls in CLASSES:
        d = SMALL / cls
        d.mkdir(parents=True, exist_ok=True)
        # clear old generated files (keep user uploads that don't match pattern)
        for i in range(n_per_class):
            img = make_small_object(cls)
            img.save(d / f"synth_{i:03d}.jpg", quality=90)
        print(f"  {cls}: {n_per_class} images")

    print("Generating satellite scenes...")
    SAT_IMG.mkdir(parents=True, exist_ok=True)
    SAT_ANN.mkdir(parents=True, exist_ok=True)
    idx = 0
    for scene in SCENES:
        for v in range(n_variants):
            img = make_satellite(scene)
            name = f"{scene['name']}_{v:02d}.jpg"
            img.save(SAT_IMG / name, quality=90)
            with open(SAT_ANN / f"{Path(name).stem}.json", "w") as f:
                json.dump({"labels": scene["labels"]}, f, indent=2)
            idx += 1
    print(f"  {idx} satellite images")
    print("Done.")


if __name__ == "__main__":
    main()
