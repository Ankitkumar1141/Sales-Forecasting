from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils import generate_covariates
from config import read_yaml
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import json
import numpy as np
import pandas as pd
import joblib


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        cfg = read_yaml().get("model_evaluation", {})
        mt_cfg = read_yaml().get("model_trainer", {})
        dt_cfg = read_yaml().get("data_transformation", {})

        self.trained_model: str = mt_cfg.get("trained_model", os.path.join("artifacts", "trained_model.joblib"))
        self.oil_model: str = mt_cfg.get("oil_model", os.path.join("artifacts", "oil_model.joblib"))
        self.covariates: str = dt_cfg.get("covariates", os.path.join("artifacts", "covariates.joblib"))
        self.testseries_data: str = dt_cfg.get("testseries_data", os.path.join("artifacts", "testseries_data.joblib"))
        self.timeseries_data: str = dt_cfg.get("timeseries_data", os.path.join("artifacts", "timeseries_data.joblib"))
        self.test_covariates: str = dt_cfg.get("test_covariates", os.path.join("artifacts", "test_covariates.joblib"))
        self.results_json: str = cfg.get("results_json", os.path.join("artifacts", "results.json"))

        # Model validation quality thresholds
        thresholds = cfg.get("thresholds", {})
        self.max_mse: float = float(thresholds.get("max_mse", 0.10))
        self.max_mae: float = float(thresholds.get("max_mae", 0.25))


class ModelEvaluation:
    def __init__(self):
        logging.info(">>> MODEL EVALUATION STARTED <<< ")
        self.modelevaluationconfig = ModelEvaluationConfig()

    def generate_predictions(self):
        """
        Evaluates the trained model by forecasting on the held-out test horizon.
        """
        logging.info("executing the generate_predictions function")
        try:
            trained_model = joblib.load(self.modelevaluationconfig.trained_model)
            oil_model = joblib.load(self.modelevaluationconfig.oil_model)
            covariates = joblib.load(self.modelevaluationconfig.covariates)
            testseries_data = joblib.load(self.modelevaluationconfig.testseries_data)
            test_covariates = joblib.load(self.modelevaluationconfig.test_covariates)
            timeseries_data = joblib.load(self.modelevaluationconfig.timeseries_data)

            logging.info("generating covariates for the next few days after the end of train data")

            # Helper for Darts version compatibility (pd_series/to_series, pd_dataframe/to_dataframe)
            def _to_series(ts):
                return ts.to_series() if hasattr(ts, "to_series") else ts.pd_series()

            def _to_df(ts):
                return ts.to_dataframe() if hasattr(ts, "to_dataframe") else ts.pd_dataframe()

            oil_forecasts = _to_series(oil_model.predict(n=len(testseries_data))).to_list()
            new_covariates = {}
            for component in testseries_data.components:
                new_covariates[component] = covariates[component].append(
                    generate_covariates(
                        horizon=len(testseries_data),
                        onpromotion=_to_df(test_covariates[component])["onpromotion"],
                        oil_forecasts=oil_forecasts,
                        is_holiday=_to_df(test_covariates[component])["is_holiday"],
                        trained_last_date=oil_model.training_series.end_time()
                    )
                )

            logging.info("forecasting for the days equal to number of test data records")

            predictions = trained_model.predict(
                n=len(testseries_data),
                series=[timeseries_data[component] for component in timeseries_data.components],
                past_covariates=[new_covariates[str(component)] for component in timeseries_data.components]
            )

            # Build DataFrame at once to prevent fragmentation warnings
            predictions_dict = {}
            for prediction in predictions:
                predictions_dict[prediction.components[0]] = _to_series(prediction).to_list()
            predictions_df = pd.DataFrame(predictions_dict)

            logging.info("returning the train data, test data and predictions for evaluation")

            return _to_df(timeseries_data), _to_df(testseries_data), predictions_df

        except Exception as e:
            logging.error(f"Error in generate_predictions: {e}")
            raise CustomException(e)

    def evaluate_predictions(self, train_data, targets, predictions):
        """
        Normalizes test data and predictions, computes metrics, gates performance
        against quality thresholds, and writes the report to JSON.
        """
        try:
            logging.info("initialising MinMaxScaler and fitting it on the train data to transform the test set and predictions")

            scaler = MinMaxScaler()
            scaler.fit(np.array(train_data))
            real_values = scaler.transform(np.array(targets))
            predicted_values = scaler.transform(np.array(predictions))

            real = []
            pred = []
            for col in range(real_values.shape[1]):
                real += list(real_values[:, col])
                pred += list(predicted_values[:, col])

            mse = float(mean_squared_error(real, pred))
            mae = float(mean_absolute_error(real, pred))
            rmse = float(np.sqrt(mse))

            # Quality Gating: Check against acceptance criteria
            passed_mse = mse <= self.modelevaluationconfig.max_mse
            passed_mae = mae <= self.modelevaluationconfig.max_mae
            status = "PASSED" if (passed_mse and passed_mae) else "FAILED_THRESHOLD"

            logging.info(f"Model Quality Gate: status={status} (MSE: {mse:.5f} vs max {self.modelevaluationconfig.max_mse}, MAE: {mae:.5f} vs max {self.modelevaluationconfig.max_mae})")

            results = {
                "Mean Squared Error (on normalised data)": mse,
                "Mean Absolute Error (on normalised data)": mae,
                "Root Mean Squared Error (on normalised data)": rmse,
                "quality_gate": {
                    "status": status,
                    "max_mse_threshold": self.modelevaluationconfig.max_mse,
                    "max_mae_threshold": self.modelevaluationconfig.max_mae,
                    "passed": passed_mse and passed_mae
                }
            }

            os.makedirs(os.path.dirname(self.modelevaluationconfig.results_json), exist_ok=True)
            with open(self.modelevaluationconfig.results_json, "w", encoding="utf-8") as jsonfile:
                json.dump(results, jsonfile, indent=3)

            logging.info("model evaluation report saved successfully to artifacts")
            return results

        except Exception as e:
            logging.error(f"Error in evaluate_predictions: {e}")
            raise CustomException(e)
