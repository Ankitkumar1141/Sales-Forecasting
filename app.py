from src.pipelines.prediction_pipeline import PredictionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel, Field
import os

# Initializing FastAPI application
app = FastAPI(
    title="Sales Forecasting API",
    description="Production-grade multivariate time-series sales forecasting using LightGBM and Darts.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware for frontend and external client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cached prediction pipeline instance
_pipeline_obj = None


def get_pipeline() -> PredictionPipeline:
    global _pipeline_obj
    if _pipeline_obj is None:
        _pipeline_obj = PredictionPipeline()
    return _pipeline_obj


# Pydantic schema with interactive examples for Swagger UI
class CovariateParams(BaseModel):
    store_nbr: int = Field(1, ge=1, description="Store number", json_schema_extra={"example": 1})
    family: str = Field("AUTOMOTIVE", description="Product family/department", json_schema_extra={"example": "AUTOMOTIVE"})
    horizon: int = Field(5, ge=1, le=30, description="Forecast horizon in days (1 to 30)", json_schema_extra={"example": 5})
    onpromotion: List[int] = Field([0, 0, 0, 0, 0], description="Promotion items per forecast day", json_schema_extra={"example": [0, 0, 0, 0, 0]})
    is_holiday: List[int] = Field([0, 0, 0, 0, 0], description="Holiday flag (0 or 1) per forecast day", json_schema_extra={"example": [0, 0, 0, 0, 0]})


class ForecastResponse(BaseModel):
    store_nbr: int
    family: str
    horizon: int
    forecasts: List[float]


@app.get("/health", tags=["Monitoring"])
@app.get("/healthz", tags=["Monitoring"])
def health_check():
    """
    Liveness & readiness probe for Docker, Kubernetes, and PaaS hosting (Render/AWS).
    """
    model_path = os.path.join("artifacts", "trained_model.joblib")
    is_ready = os.path.exists(model_path)
    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "sales-forecasting",
        "model_artifact_present": is_ready,
        "version": "1.0.0"
    }


@app.get("/info", tags=["Metadata"])
def get_info():
    """
    Returns API metadata and supported capabilities.
    """
    return {
        "title": "Sales Forecasting API",
        "architecture": "Multivariate Time Series (LightGBM + Darts)",
        "models": ["LightGBM Sales Forecaster", "LightGBM Oil Covariate Forecaster"],
        "max_forecast_horizon": 30,
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "documentation": "GET /docs"
        }
    }


@app.get("/", tags=["Forecasting"])
def root(params: Optional[CovariateParams] = None):
    """
    Root endpoint:
    - If visited in a browser without a body, returns welcome status and link to /docs.
    - If called with a JSON body, returns the generated forecasts.
    """
    if params is not None:
        return predict_forecasts(params)

    return {
        "status": "online",
        "service": "Sales Forecasting API",
        "version": "1.0.0",
        "documentation": "/docs",
        "swagger_ui": "http://127.0.0.1:8000/docs",
        "health_check": "http://127.0.0.1:8000/health",
        "message": "Visit /docs to test live predictions interactively."
    }


@app.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
@app.post("/", response_model=ForecastResponse, include_in_schema=False)
def predict_forecasts(params: CovariateParams):
    """
    Main prediction endpoint:
    Generates point forecasts for daily store sales based on product family and covariates.
    """
    try:
        pipeline = get_pipeline()
        predictions = pipeline.produce_forecasts(
            store_nbr=params.store_nbr,
            family=params.family,
            horizon=params.horizon,
            onpromotion=params.onpromotion,
            is_holiday=params.is_holiday,
        )
        return ForecastResponse(
            store_nbr=params.store_nbr,
            family=params.family.upper().strip(),
            horizon=params.horizon,
            forecasts=predictions
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal forecast error: {str(e)}")
