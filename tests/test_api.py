import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, root, health_check, get_info, predict_forecasts, CovariateParams


class TestFastAPIEndpoints(unittest.TestCase):
    """Tests for API routes."""

    VALID_PAYLOAD = CovariateParams(
        store_nbr=1,
        family="AUTOMOTIVE",
        horizon=5,
        onpromotion=[0, 0, 0, 0, 0],
        is_holiday=[0, 0, 0, 0, 0]
    )

    def test_health_check(self):
        """Test /health probe returns healthy status."""
        response = health_check()
        self.assertIn("status", response)
        self.assertIn("service", response)
        self.assertEqual(response["service"], "sales-forecasting")

    def test_info_endpoint(self):
        """Test /info endpoint returns metadata."""
        info = get_info()
        self.assertIn("title", info)
        self.assertIn("architecture", info)
        self.assertEqual(info["max_forecast_horizon"], 30)

    def test_browser_root_welcome(self):
        """Test visiting / without body returns online status."""
        res = root(None)
        self.assertEqual(res["status"], "online")
        self.assertIn("documentation", res)

    def test_predict_forecasts_valid(self):
        """Test valid forecast produces expected horizon length."""
        model_exists = os.path.exists(os.path.join("artifacts", "trained_model.joblib"))
        if not model_exists:
            self.skipTest("Model artifacts not found; skipping inference test.")

        response = predict_forecasts(self.VALID_PAYLOAD)
        self.assertEqual(response.store_nbr, 1)
        self.assertEqual(response.family, "AUTOMOTIVE")
        self.assertEqual(response.horizon, 5)
        self.assertEqual(len(response.forecasts), 5)
        self.assertTrue(all(isinstance(x, float) for x in response.forecasts))


if __name__ == "__main__":
    unittest.main()
