# SatVision Lab

**Local AI lab for satellite imagery: single-object recognition + multi-label scene annotation.**

SatVision Lab is a full-stack, offline-friendly platform that runs two complementary remote-sensing models side by side:

| Workflow | Page | Question it answers | Output |
|----------|------|---------------------|--------|
| **Object Classification** | `/classify` | *What is the primary object in this patch?* | **One** class (aircraft, ship, vehicle, building) + Top-3 alternatives |
| **Scene Annotation** | `/annotate` | *What semantic labels describe this entire scene?* | **Multiple** land-cover labels filtered by a confidence threshold |

Both models use **EfficientNet-B3 + CBAM** (via `timm`), GradCAM explainability, and a dark glassmorphism UI. Data and weights stay under `backend/data/` — no cloud database required for core use.

> **Short description (for repos / portfolios):**  
> SatVision Lab — dual-model satellite AI workbench: EfficientNet-B3+CBAM object classifier and multi-label scene annotator with GradCAM, training metrics, dataset tools, and a unified analysis pipeline.

---

## What’s included

- **Object Classification** — gold hero result, confidence ring, Top-3 bars, prediction details, GradCAM
- **Scene Annotation** — generated annotation chips, all-label score bars, threshold accept/reject, GradCAM
- **Unified Pipeline** — scene labels → sliding-window patches → high-confidence object detections only
- **Dataset Manager** — class folders + satellite images with **label filter** chips
- **Metrics** — split classifier / annotator columns, curves, confusion matrix, admin train unlock
- **Logs** — append-only prediction history
- **Admin lock** — train + dataset writes require password; viewing metrics is open

### Current lab weights (example run)

| Model | Architecture | Primary metric |
|-------|--------------|----------------|
| Classifier | EfficientNet-B3 + CBAM + MLP | ~**100%** val accuracy (4 classes) |
| Annotator | EfficientNet-B3 + CBAM + multi-label head | ~**97%** Hamming accuracy · high mAP on active labels |

Weights live in `backend/data/models/`.

---

## Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI, PyTorch, timm, albumentations, scikit-learn |
| Frontend | React 18, TypeScript, Vite, Tailwind, Recharts, Axios |
| Storage | Local folders under `backend/data/` |

---

## Requirements

- Python **3.10+**
- Node.js **18+**
- ~2 GB disk for deps + ImageNet backbone download (first run)

---

## Quick start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --host 127.0.0.1 --port 8003
```

API docs: http://127.0.0.1:8003/docs  

> The Vite proxy targets **port 8003** (`frontend/vite.config.ts`). If you change the port, update that file.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

UI: http://127.0.0.1:5173

### 3. Open the app

| Page | URL |
|------|-----|
| Dashboard | http://127.0.0.1:5173/ |
| Object Classification | http://127.0.0.1:5173/classify |
| Scene Annotation | http://127.0.0.1:5173/annotate |
| Pipeline | http://127.0.0.1:5173/pipeline |
| Dataset | http://127.0.0.1:5173/dataset |
| Metrics / Training | http://127.0.0.1:5173/training |
| Logs | http://127.0.0.1:5173/logs |

---

## Admin lock (train & dataset writes)

Training and dataset upload/delete need an admin session.

| Setting | Default |
|---------|---------|
| Password | `admin123` |
| Config file | `backend/data/admin_config.json` |
| Env override | `ADMIN_PASSWORD` |

Unlock from **Metrics** → Admin panel (password field includes show/hide).

Protected: `POST /classify/train`, `POST /annotate/train`, dataset `POST` / `DELETE`.

Anyone can still view metrics, curves, confusion matrices, and run predictions.

---

## Models

### Object classifier (single-label)

- **Input:** `3 × 128 × 128` RGB patch  
- **Backbone:** EfficientNet-B3 (`timm`, ImageNet pretrained, features)  
- **Attention:** CBAM (channel + spatial)  
- **Head:** GAP → MLP → Softmax  
- **Loss:** CrossEntropy + label smoothing  
- **Classes:** `aircraft`, `building`, `ship`, `vehicle`  
- **Weights:** `backend/data/models/small_object_classifier.pth`

### Scene annotator (multi-label)

- **Input:** `3 × 224 × 224` RGB scene  
- **Backbone:** EfficientNet-B3 + CBAM  
- **Head:** GAP → MLP → Sigmoid (per label)  
- **Loss:** BCEWithLogits + positive class weights  
- **Metric focus:** Hamming accuracy, active-label F1, mAP  
- **Default threshold:** `0.45` (UI slider live-filters annotations)  
- **Weights:** `backend/data/models/multilabel_annotator.pth`

Default environmental labels:

`water_body`, `urban_area`, `dense_forest`, `sparse_vegetation`, `agriculture`, `barren_land`, `desert`, `cloud_cover`, `flood`, `wildfire`, `snow_ice`, `wetland`

---

## Real datasets (recommended)

| Task | Source | Use |
|------|--------|-----|
| Objects | **NWPU VHR-10** | Cropped airplane / ship / vehicle / storage-tank → building |
| Scenes | **EuroSAT RGB** | Land-cover classes mapped to multi-label annotations |

```bash
cd backend

# Download (once)
curl -L -o data/raw/NWPU-VHR-10.zip \
  "https://huggingface.co/datasets/torchgeo/vhr10/resolve/main/NWPU%20VHR-10%20dataset.zip"
curl -L -o data/raw/vhr10_annotations.json \
  "https://huggingface.co/datasets/torchgeo/vhr10/resolve/main/annotations.json"
python3 -c "import torchvision; torchvision.datasets.EuroSAT(root='data/raw/eurosat_tv_download', download=True)"

# Build train + held-out test splits
python3 prepare_real_datasets.py

# Train + evaluate
python3 train_real.py
# or: python3 finish_train_eval.py
```

After prepare:

```
backend/data/datasets/   # training
backend/data/test/       # held-out test
backend/data/dataset_manifest.json
```

`seed_dataset.py` is optional for a quick synthetic smoke test.

---

## Adding your own data

### Object classes

```
backend/data/datasets/small_objects/
  aircraft/*.jpg
  building/*.jpg
  ship/*.jpg
  vehicle/*.jpg
```

Or use **Dataset → Small Object Classes** (admin unlock required to write).

### Satellite scenes

```
backend/data/datasets/satellite/images/scene_001.jpg
backend/data/datasets/satellite/annotations/scene_001.json
```

```json
{ "labels": ["water_body", "urban_area", "cloud_cover"] }
```

On the Dataset page, click a label chip to **filter** the list to matching images.

---

## Training

1. Unlock admin on **Metrics**
2. Set epochs / learning rate / batch size (defaults tuned for high accuracy)
3. Start classifier and/or annotator training (synchronous)

Or via API / CLI:

```bash
# multipart form — see /docs
# CLI helpers:
python3 train_all.py
python3 train_real.py
```

Outputs:

- `backend/data/models/small_object_classifier.pth`
- `backend/data/models/multilabel_annotator.pth`
- `backend/data/models/model_meta.json`
- `backend/data/logs/training_log.txt`

---

## Pipeline notes

The unified pipeline runs **two models**:

1. **Scene annotator** → land-cover labels for the whole image  
2. **Object classifier** → boxes only when the scene is large enough and detections pass a **high confidence + margin** filter  

Tiny EuroSAT-style tiles often correctly show scene labels (e.g. water / agriculture) and **no** object boxes — that is intentional, so water is not force-labeled as “ship.”

---

## Project layout

```
SatVision Lab/
├── README.md
├── backend/
│   ├── main.py                 # FastAPI routes
│   ├── model_service.py        # architectures, train, infer, GradCAM
│   ├── data_utils.py
│   ├── admin_auth.py
│   ├── prepare_real_datasets.py
│   ├── train_real.py
│   ├── requirements.txt
│   └── data/
│       ├── models/
│       ├── datasets/
│       ├── test/
│       ├── uploads/
│       ├── logs/
│       └── admin_config.json
└── frontend/
    ├── vite.config.ts          # proxies /api + /files → :8003
    ├── package.json
    └── src/
        ├── pages/              # Dashboard, Classify, Annotate, …
        ├── components/
        └── api/client.ts
```

---

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Status + loaded models |
| GET | `/meta` | Model metadata |
| POST | `/classify/predict` | Single-label + GradCAM + `inference_ms` |
| POST | `/classify/train` | Train classifier *(admin)* |
| GET | `/classify/status` | Accuracy, curves, confusion |
| POST | `/annotate/predict` | Multi-label + GradCAM + threshold |
| POST | `/annotate/train` | Train annotator *(admin)* |
| GET | `/annotate/status` | Accuracy, mAP, curves |
| POST | `/pipeline/analyze` | Scene labels + filtered object patches |
| GET/POST/DELETE | `/dataset/...` | Local dataset management |
| GET | `/training/logs` | Training log tail |
| GET/DELETE | `/logs` | Prediction history |
| POST | `/admin/login` | Admin unlock |

---

## Notes

- CPU by default; CUDA is used when available.
- First backbone download needs network once (`timm` / Hugging Face).
- Keep frontend proxy port aligned with the uvicorn port (**8003** by default in this lab).
- Designed as a **local teaching / demo lab**, not a production multi-tenant SaaS.
