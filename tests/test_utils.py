"""
Unit tests for pipeline utility functions.
Supports both:
  python -m unittest discover tests
  pytest tests/
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from darts import TimeSeries
from pandas import Timestamp
from src.utils import generate_covariates


class TestGenerateCovariates(unittest.TestCase):
    """Tests for generate_covariates utility function."""

    def test_valid_input(self):
        """Test that valid inputs produce a Darts TimeSeries object."""
        result = generate_covariates(
            horizon=5,
            onpromotion=[1, 0, 1, 0, 1],
            oil_forecasts=[70.0, 71.0, 72.0, 73.0, 74.0, 75.0],
            is_holiday=[0, 0, 1, 0, 0],
            trained_last_date=Timestamp("2017-08-15")
        )
        self.assertIsInstance(result, TimeSeries)
        self.assertEqual(len(result), 5)

    def test_horizon_exceeds_max(self):
        """Test that horizon > 30 returns an error string."""
        result = generate_covariates(
            horizon=35,
            onpromotion=[0] * 35,
            oil_forecasts=[70.0] * 35,
            is_holiday=[0] * 35,
            trained_last_date=Timestamp("2017-08-15")
        )
        self.assertEqual(result, "Forecast horizon cannot be greater than 30")

    def test_zero_horizon(self):
        """Test that horizon <= 0 returns an error string."""
        result = generate_covariates(
            horizon=0,
            onpromotion=[],
            oil_forecasts=[],
            is_holiday=[],
            trained_last_date=Timestamp("2017-08-15")
        )
        self.assertEqual(result, "Forecast horizon must be positive")

    def test_length_mismatch(self):
        """Test that mismatched lengths return an error string."""
        result = generate_covariates(
            horizon=5,
            onpromotion=[1, 0],  # too short
            oil_forecasts=[70.0] * 5,
            is_holiday=[0, 0, 1, 0, 0],
            trained_last_date=Timestamp("2017-08-15")
        )
        self.assertEqual(result, "Length mismatch")


if __name__ == "__main__":
    unittest.main()
