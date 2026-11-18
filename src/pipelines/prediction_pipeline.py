from src.utils import generate_covariates
from src.utils.exception import CustomException
from config import read_yaml
import joblib
import os
from typing import List
from dataclasses import dataclass


@dataclass
class PredictionPipelineConfig:
    def __init__(self):
        cfg = read_yaml().get("prediction", {})
        mt_cfg = read_yaml().get("model_trainer", {})
        dt_cfg = read_yaml().get("data_transformation", {})

        self.oil_model_path: str = mt_cfg.get("oil_model", os.path.join("artifacts", "oil_model.joblib"))
        self.trained_model_path: str = mt_cfg.get("trained_model", os.path.join("artifacts", "trained_model.joblib"))
        self.covariates_path: str = dt_cfg.get("covariates", os.path.join("artifacts", "covariates.joblib"))
        self.timeseries_data_path: str = dt_cfg.get("timeseries_data", os.path.join("artifacts", "timeseries_data.joblib"))
        self.max_forecast_horizon: int = int(cfg.get("max_forecast_horizon", 30))


class PredictionPipeline:
    def __init__(self):
        self.predictionpipelineconfig = PredictionPipelineConfig()
        self._oil_model = None
        self._trained_model = None
        self._covariates = None
        self._timeseries_data = None

    def _load_artifacts(self):
        """Lazy load artifacts once into memory for fast inference."""
        if self._oil_model is None:
            if not os.path.exists(self.predictionpipelineconfig.trained_model_path):
                raise FileNotFoundError(
                    f"Model artifact not found at '{self.predictionpipelineconfig.trained_model_path}'. "
                    "Run the training pipeline first: python src/pipelines/train_pipeline.py"
                )
            self._oil_model = joblib.load(self.predictionpipelineconfig.oil_model_path)
            self._trained_model = joblib.load(self.predictionpipelineconfig.trained_model_path)
            self._covariates = joblib.load(self.predictionpipelineconfig.covariates_path)
            self._timeseries_data = joblib.load(self.predictionpipelineconfig.timeseries_data_path)

    def produce_forecasts(
        self,
        store_nbr: int,
        family: str,
        horizon: int,
        onpromotion: List[int],
        is_holiday: List[int],
    ) -> List[float]:
        """
        Produces sales forecasts for the given store and product family over the specified horizon.
        Raises ValueError for invalid parameters.
        """
        max_horizon = self.predictionpipelineconfig.max_forecast_horizon
        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be positive. Received: {horizon}")
        if horizon > max_horizon:
            raise ValueError(f"Forecast horizon cannot exceed {max_horizon} days. Received: {horizon}")
        if len(onpromotion) < horizon:
            raise ValueError(f"Length of onpromotion list ({len(onpromotion)}) is shorter than horizon ({horizon})")
        if len(is_holiday) < horizon:
            raise ValueError(f"Length of is_holiday list ({len(is_holiday)}) is shorter than horizon ({horizon})")

        try:
            self._load_artifacts()

            series_name = str((store_nbr, family.upper().strip()))
            if series_name not in self._covariates.keys():
                available = [eval(k) for k in list(self._covariates.keys())[:5]]
                raise ValueError(
                    f"Combination store_nbr={store_nbr}, family='{family}' not found in trained series. "
                    f"Examples of valid pairs: {available}..."
                )

            # Helper for Darts version compatibility (pd_series/to_series)
            def _to_series(ts):
                return ts.to_series() if hasattr(ts, "to_series") else ts.pd_series()

            # Generating oil forecasts for 30 days
            oil_forecasts = _to_series(self._oil_model.predict(n=30)).to_list()

            # Generating new past covariates for final model
            covariate = generate_covariates(
                horizon=horizon,
                onpromotion=onpromotion,
                oil_forecasts=oil_forecasts,
                is_holiday=is_holiday,
                trained_last_date=self._oil_model.training_series.end_time()
            )

            if isinstance(covariate, str):
                raise ValueError(covariate)

            new_covariates = self._covariates[series_name].append(covariate)

            # Generating sales predictions for the supplied forecast horizon
            predictions = self._trained_model.predict(
                n=horizon,
                series=self._timeseries_data[series_name],
                past_covariates=new_covariates
            )

            return [round(float(x), 2) for x in _to_series(predictions).to_list()]

        except ValueError:
            raise
        except Exception as e:
            raise CustomException(e)
