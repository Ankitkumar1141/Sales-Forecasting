import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import read_yaml
from src.pipelines.prediction_pipeline import PredictionPipeline


class TestPipelineConfig(unittest.TestCase):
    """Tests for YAML config integrity."""

    def test_config_keys_present(self):
        """Ensure all required pipeline sections exist in config.yaml."""
        cfg = read_yaml()
        self.assertIn("data_ingestion", cfg)
        self.assertIn("data_transformation", cfg)
        self.assertIn("model_trainer", cfg)
        self.assertIn("model_evaluation", cfg)
        self.assertIn("prediction", cfg)
        self.assertIn("api", cfg)

    def test_evaluation_thresholds(self):
        """Ensure model quality gate thresholds are configured."""
        cfg = read_yaml()
        eval_cfg = cfg.get("model_evaluation", {})
        thresholds = eval_cfg.get("thresholds", {})
        self.assertIn("max_mse", thresholds)
        self.assertIn("max_mae", thresholds)
        self.assertGreater(thresholds["max_mse"], 0)

    def test_prediction_input_validation(self):
        """Ensure PredictionPipeline rejects invalid horizons."""
        pipeline = PredictionPipeline()
        with self.assertRaises(ValueError):
            pipeline.produce_forecasts(
                store_nbr=1,
                family="AUTOMOTIVE",
                horizon=0,  # invalid horizon
                onpromotion=[],
                is_holiday=[]
            )

        with self.assertRaises(ValueError):
            pipeline.produce_forecasts(
                store_nbr=1,
                family="AUTOMOTIVE",
                horizon=50,  # exceeds max 30
                onpromotion=[0] * 50,
                is_holiday=[0] * 50
            )


if __name__ == "__main__":
    unittest.main()
