from src.utils.exception import CustomException
from src.utils.logger import logging
from config import read_yaml
from dataclasses import dataclass
from darts.models.forecasting.lgbm import LightGBMModel
import joblib
import os


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        cfg = read_yaml().get("model_trainer", {})
        self.timeseries_data: str = cfg.get("timeseries_data", os.path.join("artifacts", "timeseries_data.joblib"))
        self.covariates: str = cfg.get("covariates", os.path.join("artifacts", "covariates.joblib"))
        self.oil_model: str = cfg.get("oil_model", os.path.join("artifacts", "oil_model.joblib"))
        self.trained_model: str = cfg.get("trained_model", os.path.join("artifacts", "trained_model.joblib"))

        # Hyperparameters from config
        self.oil_model_params: dict = cfg.get("oil_model_params", {
            "lags": 25,
            "output_chunk_length": 1,
            "n_estimators": 500,
            "verbosity": 0
        })
        self.sales_model_params: dict = cfg.get("sales_model_params", {
            "lags": [-1, -2, -6, -7, -8, -13, -14, -15, -20, -21, -27, -28, -35, -42, -49, -56, -63],
            "lags_past_covariates": [-1, -2, -6, -7, -8, -13, -14, -15, -20, -21, -27, -28, -35],
            "output_chunk_length": 1,
            "n_estimators": 1000,
            "verbosity": 0
        })


class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig = ModelTrainerConfig()
        logging.info(">>> MODEL TRAINER STARTED <<<")

    def train_model(self):
        """
        Trains the oil forecasting model and the primary LightGBM sales forecasting model.
        Hyperparameters and artifact paths are loaded from config/config.yaml.
        """
        logging.info("executing train_model function")
        try:
            timeseries_data = joblib.load(self.modeltrainerconfig.timeseries_data)
            covariates = joblib.load(self.modeltrainerconfig.covariates)

            logging.info("training LightGBM Model for producing oil forecasts")
            series = covariates[str((1, "AUTOMOTIVE"))]["dcoilwtico"]
            oil_model = LightGBMModel(**self.modeltrainerconfig.oil_model_params)
            oil_model.fit(series=series)

            logging.info("creating the final LightGBM Model and training it to forecast sales")
            model = LightGBMModel(**self.modeltrainerconfig.sales_model_params)

            logging.info("fitting the LightGBM Model for forecasting sales")
            model.fit(
                series=[timeseries_data[component] for component in timeseries_data.components],
                past_covariates=[covariates[cov] for cov in timeseries_data.components]
            )

            logging.info("saving the trained models to artifacts")
            os.makedirs(os.path.dirname(self.modeltrainerconfig.oil_model), exist_ok=True)
            joblib.dump(oil_model, self.modeltrainerconfig.oil_model)
            joblib.dump(model, self.modeltrainerconfig.trained_model)

            logging.info("models saved successfully")
            logging.info(">>> MODEL TRAINER COMPLETED <<<")

        except Exception as e:
            logging.error(f"Error in train_model: {e}")
            raise CustomException(e)
