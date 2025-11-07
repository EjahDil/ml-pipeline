import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import os
from unittest.mock import patch


from data_pipeline.modules.drift_detector import DriftDetector

def run_detect(detector, current_path, ref_path=None):
    """Run the async detect_drift() in the current event loop."""
    import asyncio
    return asyncio.run(detector.detect_drift(current_path, ref_path))


# ----------------------------------------------------------------------
# 3. Fixtures (same data as before)
# ----------------------------------------------------------------------
@pytest.fixture
def sample_config():
    return {
        "enabled": True,
        "reference_data_path": "ref.parquet",
        "metrics": ["psi", "ks", "wasserstein"],
        "thresholds": {"psi": 0.2, "ks": 0.1, "wasserstein": 0.15},
        "categorical_columns": ["category"],
        "numerical_columns": ["value"],
    }


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "value": np.random.normal(0, 1, 100),
            "category": np.random.choice(["A", "B", "C"], size=100),
        }
    )


@pytest.fixture
def current_no_drift():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "value": np.random.normal(0, 1, 80),
            "category": np.random.choice(["A", "B", "C"], size=80),
        }
    )


@pytest.fixture
def current_with_drift():
    np.random.seed(123)
    return pd.DataFrame(
        {
            "value": np.random.normal(5, 1, 80),          # shifted mean
            "category": np.random.choice(["X", "Y"], size=80),  # new cats
        }
    )


# ----------------------------------------------------------------------
# 4. Tests – no @pytest.mark.asyncio, no async def
# ----------------------------------------------------------------------
def test_no_drift(sample_config, reference_data, current_no_drift):
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.parquet")
        cur_path = os.path.join(tmp, "cur.parquet")
        reference_data.to_parquet(ref_path)
        current_no_drift.to_parquet(cur_path)

        cfg = sample_config.copy()
        cfg["reference_data_path"] = ref_path
        detector = DriftDetector(cfg)

        result = run_detect(detector, cur_path)

        assert result["drift_detected"] is False
        assert len(result["alerts"]) == 0
        assert "value" in result["metrics"]
        assert "category" in result["metrics"]


def test_with_drift(sample_config, reference_data, current_with_drift):
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.parquet")
        cur_path = os.path.join(tmp, "cur.parquet")
        reference_data.to_parquet(ref_path)
        current_with_drift.to_parquet(cur_path)

        cfg = sample_config.copy()
        cfg["reference_data_path"] = ref_path
        detector = DriftDetector(cfg)

        result = run_detect(detector, cur_path)

        assert result["drift_detected"] is True
        assert len(result["alerts"]) >= 2
        assert any("value" in a for a in result["alerts"])
        assert any("category" in a for a in result["alerts"])


def test_reference_created_when_missing(sample_config, current_no_drift):
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.parquet")
        cur_path = os.path.join(tmp, "cur.parquet")
        current_no_drift.to_parquet(cur_path)

        cfg = sample_config.copy()
        cfg["reference_data_path"] = ref_path   # does NOT exist yet
        detector = DriftDetector(cfg)

        result = run_detect(detector, cur_path)

        assert Path(ref_path).exists()
        assert result["drift_detected"] is False
        assert "Reference data created" in result.get("message", "")


def test_disabled_detector(sample_config):
    cfg = sample_config.copy()
    cfg["enabled"] = False
    detector = DriftDetector(cfg)

    result = run_detect(detector, "dummy.csv")
    assert result["enabled"] is False
    assert result["drift_detected"] is False


def test_missing_column(sample_config, reference_data):
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.parquet")
        cur_path = os.path.join(tmp, "cur.parquet")

        ref_df = reference_data.copy()
        ref_df["extra"] = 1
        ref_df.to_parquet(ref_path)
        reference_data.to_parquet(cur_path)   # no 'extra'

        cfg = sample_config.copy()
        cfg["reference_data_path"] = ref_path
        cfg["numerical_columns"] = ["value", "extra"]
        detector = DriftDetector(cfg)

        with patch("logging.Logger.warning") as mock_warn:
            run_detect(detector, cur_path)
            mock_warn.assert_called_with("Column extra missing in current data")