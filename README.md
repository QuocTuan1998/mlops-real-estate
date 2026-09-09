# MLOps Real Estate Price Prediction Pipeline

An end-to-end MLOps project demonstrating the full machine learning lifecycle for predicting real estate prices in Vietnam — from data ingestion and training, through MLflow-based experiment tracking and model registry promotion, to serving predictions via a FastAPI inference service, containerized with Docker/Podman and automated with CI/CD.

## Project Structure

```
mlops-real-estate
├── api/                        # FastAPI inference service
│   ├── main.py                  # API endpoints (/health, /model, /predict, /reload-model)
│   ├── model_loader.py          # Loads the current Production model from MLflow registry
│   └── schemas.py                # Pydantic request/response models
├── configs/
│   └── config.yaml               # Data, training, MLflow, and promotion settings
├── data/
│   └── raw/
│       └── real_estate_sample.csv
├── src/
│   ├── data/
│   │   └── preprocess.py          # Load & validate raw data
│   ├── features/
│   │   └── build_features.py      # Feature engineering
│   ├── models/
│   │   ├── train.py                 # Model training (train/test split)
│   │   ├── predict.py               # Prediction helper used by the API
│   │   └── evaluate.py              # Metric computation (MAE, RMSE, R2)
│   ├── training/
│   │   └── train_pipeline.py         # Standalone training + MLflow logging
│   ├── pipeline/
│   │   └── update_pipeline.py         # Candidate vs. Production comparison & promotion
│   └── utils/
│       └── config.py                   # YAML config loader
├── tests/                          # Unit tests (pytest)
├── notebooks/
│   └── exploration.ipynb            # Exploratory data analysis
├── mlruns/                          # MLflow experiment/artifact store (gitignored)
├── mlflow.db                         # MLflow SQLite tracking backend (gitignored)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── setup.py
├── .github/workflows/ci-cd.yml       # Lint → test → build → smoke test → push
└── README.md
```

## Dataset

`data/raw/real_estate_sample.csv` contains sample property listings across major Vietnamese cities/provinces (Ha Noi, Ho Chi Minh, Da Nang, Hai Phong, Can Tho, Nha Trang, Hue, Binh Duong, Vung Tau, Quang Ninh, Dong Nai, Long An), with features:

| Feature | Description |
|---|---|
| `area` | Property area (m²) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `floor` | Floor number |
| `property_age` | Age of the property (years) |
| `location` | City/Province |
| `property_type` | Apartment, House, Condo, Townhouse |
| `price` | Target — price in VND |

## MLOps Lifecycle

1. **Data Preprocessing** — `src/data/preprocess.py` loads and validates raw data against the schema in `configs/config.yaml`.
2. **Model Training** — `src/training/train_pipeline.py` trains a model, logs params/metrics/artifacts to MLflow, and registers it in the Model Registry.
3. **Model Evaluation** — `src/models/evaluate.py` computes MAE, RMSE, and R².
4. **Candidate vs. Production Promotion** — `src/pipeline/update_pipeline.py`:
   - Trains a new candidate model.
   - Evaluates it on a held-out test set.
   - Loads the current Production model (if any) and evaluates it on the **same** test set.
   - Promotes the candidate to Production only if it beats the current Production model by a configurable threshold (`configs/config.yaml` → `promotion.min_improvement`).
5. **Serving** — `api/main.py` (FastAPI) loads the current Production model at startup and exposes:
   - `GET /health` — liveness check
   - `GET /model` — metadata about the loaded model (name, version, stage)
   - `POST /predict` — price prediction for a single property
   - `POST /reload-model` — hot-reload the Production model after a new promotion, without restarting the service
6. **Monitoring & Maintenance** — Re-run `update_pipeline.py` periodically (or via CI/CD) as new data arrives; call `/reload-model` to pick up newly promoted versions.

## Dataset Versioning with DVC

Raw housing data is tracked with **DVC**, not committed directly to Git — only the
lightweight `.dvc` pointer file (`data/raw/housing.csv.dvc`) is versioned in git.

### Setup (one-time)

```bash
pip install dvc
dvc init
mkdir -p dvc_storage
dvc remote add -d localstorage dvc_storage
```

### Track the dataset

```bash
dvc add data/raw/housing.csv
git add data/raw/housing.csv.dvc data/raw/.gitignore
git commit -m "Track housing.csv with DVC (dataset v1)"
dvc push
```

### Adding a new dataset version

```bash
# Overwrite data/raw/housing.csv with the new/updated data, then:
dvc add data/raw/housing.csv
git add data/raw/housing.csv.dvc
git commit -m "Update housing dataset to v2"
dvc push
git push
```

Pushing the updated `housing.csv.dvc` pointer to `main` automatically triggers
`.github/workflows/ml-retraining.yml`, which pulls the new dataset version and
runs the full retrain → evaluate → promote/reject pipeline.

### Reproducing a specific model's training run

Every MLflow run tagged by `update_pipeline.py` records:
- `git_commit` — the exact code version used
- `dataset_version` — the DVC md5 hash of the dataset used
- `model_version` — `candidate` or `production`

To reproduce Model v1 exactly:

```bash
git checkout <git_commit_from_mlflow_run>
dvc checkout   # restores the exact dataset version pinned by that commit's .dvc file
python -m src.pipeline.update_pipeline
```

## Why the retraining workflow triggers only on `.dvc` changes

`.github/workflows/ml-retraining.yml` is scoped to:

```yaml
on:
  push:
    paths:
      - "data/raw/housing.csv.dvc"
```

This ensures retraining is triggered **only** when the dataset itself changes —
not on unrelated code changes (API tweaks, README edits, etc.). This keeps CI
fast and makes the causal link explicit: *new data in → new model evaluated out*,
which is the core principle this project demonstrates.

## Setup Instructions

### 1. Clone and install

```bash
git clone <repository-url>
cd mlops-real-estate
pip install -r requirements.txt
```

### 2. Configure MLflow tracking

This project uses a SQLite-backed MLflow tracking store (required for the Model Registry). Configured in `configs/config.yaml`:

```yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "real-estate-price-prediction"
  registry_model_name: "real_estate_price_model"
```

### 3. Train the initial model

```bash
python -m src.training.train_pipeline
```

### 4. Run the candidate-vs-production promotion pipeline

```bash
python -m src.pipeline.update_pipeline
```

The first run auto-promotes the candidate (no existing Production model to compare against). Subsequent runs compare against the current Production model and promote only on improvement.

### 5. Inspect experiments in the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` to view runs, metrics, and the Model Registry (check which version holds the `Production` stage / `production` alias).

### 6. Serve predictions via FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

Test it:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"area":80,"bedrooms":3,"bathrooms":2,"floor":10,"property_age":5,"location":"Ha Noi","property_type":"Apartment"}'
```

### 7. Run tests

```bash
pytest -v
```

## Docker / Podman

Build and run the inference service in a container:

```bash
docker build -t real-estate-price-api:latest .
docker run -p 8000:8000 real-estate-price-api:latest
```

Podman users, disable Git Bash path mangling on Windows when mounting volumes:

```bash
MSYS_NO_PATHCONV=1 podman run -d --name real-estate-api -p 8000:8000 real-estate-price-api:latest
```

> **Note:** `mlflow.db` and `mlruns/` are excluded from the image via `.dockerignore`. Train and promote a model **inside** the running container so MLflow artifact paths match the container's filesystem:
> ```bash
> podman exec real-estate-api python -m src.training.train_pipeline
> podman exec real-estate-api python -m src.pipeline.update_pipeline
> curl -X POST http://localhost:8000/reload-model
> ```

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push/PR to `main`:

1. **Test** — install dependencies, run `pytest`.
2. **Build & Push** (main branch only) — build the Docker image, smoke-test `/health`, then push to Docker Hub tagged with `latest` and the commit SHA.

Configure `DOCKER_USERNAME` and `DOCKER_PASSWORD` as GitHub repository secrets to enable image publishing.
