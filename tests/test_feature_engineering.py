import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.feature_engineering import FeatureEngineering

@pytest.fixture
def sample_config():

    return {
        "features": {
            "numerical": ["Age", "Income"],
            "categorical": ["Gender", "City"],
            "selected_features": ["Age", "Income", "Gender", "City"]
        },
        "encoding": {
            "Gender": {
                "strategy": "ordinal",
                "categories": ["Male", "Female"]
            },
            "City": {
                "strategy": "onehot",
                "categories": ["London", "Paris"]
            }
        },
        "preprocessing": {
            "scaler": "standard"
        }
    }


@pytest.fixture
def sample_data():

    return pd.DataFrame({
        "Age": [25, 40, 35],
        "Income": [40000, 80000, 60000],
        "Gender": ["Male", "Female", "Female"],
        "City": ["London", "Paris", "London"],
        "InboundCalls": [5, 2, 4],
        "OutboundCalls": [3, 6, 1]
    })


def test_fit_and_transform(sample_config, sample_data):
    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)

    transformed = fe.transform(sample_data)
    assert isinstance(transformed, pd.DataFrame)



def test_invalid_config_raises(sample_data):

    bad_config = {
        "features": {"numerical": ["Age"], "categorical": ["Color"]},
        "encoding": {"Color": {"strategy": "ordinal"}}
    }
    with pytest.raises(ValueError, match="Missing 'categories'"):
        FeatureEngineering(bad_config).fit(sample_data)


def test_scaling_modes(sample_config, sample_data):

    for scaler in ["standard", "minmax", "robust"]:
        config = sample_config.copy()
        config["preprocessing"]["scaler"] = scaler
        fe = FeatureEngineering(config)
        fe.fit(sample_data)
        transformed = fe.transform(sample_data)
        assert not transformed.isna().any().any(), f"{scaler} introduced NaNs"



def test_save_and_load(tmp_path, sample_config, sample_data):
    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)
    fe.save(tmp_path)

    loaded = FeatureEngineering.load(tmp_path)
    assert isinstance(loaded, FeatureEngineering)

    assert loaded._config_hash == fe._config_hash
    assert loaded.numerical == fe.numerical

    out = loaded.transform(sample_data)
    assert isinstance(out, pd.DataFrame)
    assert not out.empty
