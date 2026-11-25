# 🏪 Sales Forecasting — Industry-Grade ML Pipeline

> **A production-grade, multivariate time series forecasting system** built with Python 3.12, Darts, LightGBM, and FastAPI. Includes automated quality gates, dual logging, robust input validation, and containerized CI/CD.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Non--Root%20Secure-blue?logo=docker)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Test%20Gated-orange?logo=githubactions)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents
- [Architecture & Pipeline Flow](#architecture--pipeline-flow)
- [Industry-Grade Features](#industry-grade-features)
- [Model Performance & Quality Gates](#model-performance--quality-gates)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Running the Training Pipeline](#running-the-training-pipeline)
- [Automated Testing](#automated-testing)
- [API Usage & Endpoints](#api-usage--endpoints)
- [Production Deployment](#production-deployment)
- [License](#license)

---

## Architecture & Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION                                    │
│  Option A: Local CSVs (data/)  ──┐                                          │
│  Option B: PostgreSQL DB       ──┴─► artifacts/raw_data.csv, oil, stores    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                        DATA TRANSFORMATION                                  │
│  • Missing Date Interpolation       • Hampel Outlier Filter                 │
│  • Holiday Calendar Integration     • Variance Threshold Feature Selection  │
│  • 15-Day Train/Test Split          • Darts TimeSeries & Covariates Objects │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                        MODEL TRAINING (LightGBM)                            │
│  • Model 1: Oil Price Covariate Forecaster (lags=25, estimators=500)        │
│  • Model 2: Sales Forecaster (lags=17, past_covs=11, estimators=1000)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                   MODEL EVALUATION & QUALITY GATE                           │
│  • Out-of-Sample Predictions        • Quality Gate: Max MSE <= 0.10         │
│  • Metrics: MSE, MAE, RMSE          • Gating Status: PASSED / REJECTED      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                         SERVING & INFERENCE                                 │
│  • FastAPI REST API                 • Docker Non-Root Container             │
│  • Swagger UI (/docs)               • Liveness / Readiness Probe (/health)  │
│  • Point & Multi-Day Predictions    • CORS Enabled Middleware               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Industry-Grade Features

1. **Decoupled Configuration Management:**
   - All pipeline parameters, hyperparameters, and directory paths are maintained in [`config/config.yaml`](config/config.yaml) and loaded via safe config readers.
2. **Quality Gating in Model Evaluation:**
   - Automatically gates new models against strict tolerance thresholds (`max_mse: 0.10`, `max_mae: 0.25`).
3. **Dual-Handler Observability:**
   - Structured logging writes concurrently to persistent log files (`logs/running_logs.log`) and standard output (`sys.stdout`) for Docker/Kubernetes container log collectors.
4. **Resilient Data Ingestion with Automatic Fallback:**
   - Seamlessly ingests from local CSVs in `data/` or remote PostgreSQL databases with clear guidance.
5. **Secure Containerization:**
   - Dockerfile leverages multi-layer pip caching, non-root user execution (`appuser`), and container healthcheck probes (`/health`).
6. **Automated CI/CD with Quality Gates:**
   - GitHub Actions enforces linting and full test suite passes *prior* to image compilation and deployment.

---

## Model Performance & Quality Gates

| Metric | Target Threshold | Model Performance (Normalised) | Quality Gate Status |
|---|:---:|:---:|:---:|
| **Mean Squared Error (MSE)** | $\le 0.10$ | **0.04331** | ✅ **PASSED** |
| **Mean Absolute Error (MAE)** | $\le 0.25$ | **0.11309** | ✅ **PASSED** |
| **Root Mean Squared Error (RMSE)** | — | **0.20811** | ✅ **PASSED** |

*Evaluated on a 15-day hold-out cross-validation window across all 1,782 store-family time series.*

---

## Project Structure

```
├── .github/
│   └── workflows/
│       └── ci-cd.yaml          # Test-gated CI/CD pipeline
│
├── artifacts/                  # Model & data artifacts (DVC-tracked)
│   ├── raw_data.csv.dvc        # DVC pointer for raw dataset
│   ├── covariates.joblib       # Processed past covariates
│   ├── timeseries_data.joblib  # Historical sales series
│   ├── oil_model.joblib        # Trained LightGBM oil price model
│   ├── trained_model.joblib    # Trained LightGBM store sales model
│   └── results.json            # Model evaluation report & quality gate
│
├── config/
│   ├── __init__.py             # Config reader module
│   └── config.yaml             # Centralized pipeline configuration
│
├── data/                       # Local raw data directory (.gitignore protected)
│   ├── train.csv
│   ├── oil.csv
│   ├── stores.csv
│   └── holidays_events.csv
│
├── db/                         # PostgreSQL database schemas
│   ├── schema.sql              # Table definitions & indexes
│   └── seed_from_csv.sql       # \copy seeding script
│
├── logs/                       # Application logs
│   └── running_logs.log
│
├── notebooks/                  # EDA & monitoring notebooks
│   ├── Store_sales_prediction.ipynb
│   └── Model_Monitoring.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   # Resilient data loading (Local / DB)
│   │   ├── data_transformation.py # Feature engineering & Darts TimeSeries
│   │   ├── model_trainer.py    # LightGBM training (hyperparams from config)
│   │   └── model_evaluation.py # Gating, metrics & report generation
│   │
│   ├── pipelines/
│   │   ├── train_pipeline.py   # Full pipeline orchestration
│   │   └── prediction_pipeline.py # Inference pipeline with input validation
│   │
│   └── utils/
│       ├── __init__.py         # generate_covariates helper
│       ├── exception.py        # Traceback-aware CustomException
│       └── logger.py           # Dual File + Stdout logger
│
├── tests/
│   ├── test_utils.py           # Unit tests for utility functions
│   ├── test_pipeline.py        # Config & validation tests
│   └── test_api.py             # FastAPI route integration tests
│
├── app.py                      # Production FastAPI server
├── Dockerfile                  # Non-root secure Docker container
├── make.bat                    # Windows command wrapper
├── Makefile                    # Linux/macOS command wrapper
├── pyproject.toml              # Modern Python 3.12 packaging metadata
├── setup.py                    # Package setup script
├── requirements.txt            # Pinned Python dependencies
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12+ |
| **Time Series Framework** | Darts (TimeSeries) |
| **Forecasting Algorithm** | LightGBM |
| **Preprocessing & Outliers** | Hampel Filter, Scikit-learn, Pandas, NumPy |
| **API Framework** | FastAPI + Uvicorn |
| **Storage Options** | Local File System or PostgreSQL (Supabase / Neon / Railway) |
| **Containerization** | Docker (Non-root, Health Checked) |
| **CI/CD** | GitHub Actions |
| **Quality Monitoring** | Evidently AI (Data & Target Drift) |

---

## Getting Started

### 1. Clone the Project
```bash
git clone https://github.com/Ankitkumar1141/Sales-Forecasting.git
cd Sales-Forecasting
```

### 2. Set Up Virtual Environment (Python 3.12)
```bash
# Windows
python -m venv sales_venv
sales_venv\Scripts\activate

# Linux / macOS
python3.12 -m venv sales_venv
source sales_venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Training Pipeline
```bash
python src/pipelines/train_pipeline.py
# or Windows
make train
```

The pipeline executes through all 4 stages:
1. **Data Ingestion** ➔ Detects and validates data.
2. **Data Transformation** ➔ Missing value interpolation, holiday joins, Hampel filtering, and Darts series conversion.
3. **Model Training** ➔ Trains oil and sales models with parameters from `config.yaml`.
4. **Model Evaluation** ➔ Computes metrics and gates performance against quality thresholds.

---

## Automated Testing

Run the full test suite with either standard Python or pytest:

```bash
# Standard Python unittest (zero external tools required)
python -m unittest discover -s tests -p "test_*.py" -v

# Or using pytest
pytest tests/ -v
```

All 11 unit and integration tests validate:
- Covariate time series generation & boundary conditions
- Horizon limits and array length mismatch validation
- YAML configuration completeness & quality gate thresholds
- API endpoints (`/health`, `/info`, `/`, `/predict`)

---

## API Usage & Endpoints

### Start the Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# or Windows
make serve
```

### Available Endpoints

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/` | Welcome portal with links to documentation |
| `GET` | `/health` | Container liveness & readiness probe |
| `GET` | `/info` | Model metadata, architecture, and capabilities |
| `GET` | `/docs` | Interactive Swagger UI documentation |
| `POST` | `/predict` | Production sales forecasting endpoint |

### Example Request (`POST /predict`)
```json
{
  "store_nbr": 1,
  "family": "AUTOMOTIVE",
  "horizon": 5,
  "onpromotion": [0, 0, 0, 0, 0],
  "is_holiday": [0, 0, 0, 0, 0]
}
```

### Example Response
```json
{
  "store_nbr": 1,
  "family": "AUTOMOTIVE",
  "horizon": 5,
  "forecasts": [5.52, 4.27, 4.45, 3.82, 3.40]
}
```

---

## Production Deployment

### Docker Build & Run
```bash
# Build the secure non-root Docker container
docker build -t store-sales-api:latest .

# Run the container
docker run -p 8000:8000 store-sales-api:latest
```

Check container health:
```bash
curl http://localhost:8000/health
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
