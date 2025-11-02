import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
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

    assert "TotalCalls" in fe.numerical

    assert abs(transformed["Age"].mean()) < 1e-6

    assert any(c.startswith("City_") for c in transformed.columns)

    names = fe.get_feature_names_out()
    assert all(name in transformed.columns for name in names)


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


def test_ordinal_and_onehot_encoding_behavior(sample_config, sample_data):

    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)
    transformed = fe.transform(sample_data)

    assert np.issubdtype(transformed["Gender"].dtype, np.number)

    onehot_cols = [c for c in transformed.columns if c.startswith("City_")]
    assert len(onehot_cols) == len(sample_config["encoding"]["City"]["categories"]) + 1


def test_inverse_transform_roundtrip(sample_config, sample_data):
    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)
    transformed = fe.transform(sample_data)
    restored = fe.inverse_transform(transformed)

    np.testing.assert_allclose(restored["Age"], sample_data["Age"], rtol=1e-5)
    np.testing.assert_allclose(restored["Income"], sample_data["Income"], rtol=1e-5)

    expected_categories = set(list(sample_data["Gender"].unique()) + ["<UNK>"])
    assert set(restored["Gender"].unique()) <= expected_categories



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
