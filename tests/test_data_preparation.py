import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import logging
from datetime import datetime, timezone
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.pre_processing import DataPreparation

SAMPLE_CONFIG = """
data:
  test_size: 0.2
  random_state: 42
preprocessing:
  handle_missing: mean
  handle_outliers: true
features:
  numerical: ['num1', 'num2']
  target: Chrun
"""

SAMPLE_DF = pd.DataFrame({
    'num1': [1, 2, np.nan, 4, 100],
    'num2': [10, 20, 30, 40, 50],
    'cat': ['A', 'B', np.nan, 'A', 'B'],
    'target': [0, 1, 0, 1, 0]
})

@pytest.fixture
def mock_config_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SAMPLE_CONFIG)
    return str(config_path)

@pytest.fixture
def data_prep(mock_config_file):
    return DataPreparation(mock_config_file)

def test_handle_missing_values_mean(data_prep):
    df = SAMPLE_DF.copy()
    imputed_df = data_prep.handle_missing_values(df)
    assert imputed_df['num1'].isnull().sum() == 0
    assert imputed_df['num1'][2] == pytest.approx((1 + 2 + 4 + 100) / 4)  # Mean
    assert imputed_df['cat'][2] == 'A'

def test_handle_missing_values_median(data_prep):
    data_prep.preprocessing_config['handle_missing'] = 'median'
    df = SAMPLE_DF.copy()
    imputed_df = data_prep.handle_missing_values(df)
    assert imputed_df['num1'][2] == np.median([1, 2, 4, 100])  # 3.0

def test_handle_outliers(data_prep):
    df = SAMPLE_DF.copy()
    df = data_prep.handle_missing_values(df)
    outlier_df = data_prep.handle_outliers(df, ['num1', 'num2'])

    assert outlier_df['num1'][4] == pytest.approx(63.875)


@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('pipelines.pre_processing.datetime')  # patch the module's datetime object
def test_save_processed_data(mock_datetime_module, mock_to_csv, mock_makedirs, data_prep):
    fake_dt = datetime(2023, 1, 1, 0, 0, 0)
    mock_datetime_module.now.return_value = fake_dt
    mock_datetime_module.utcnow.return_value = fake_dt

   
    mock_datetime_module.strftime = datetime.strftime

    train_df = SAMPLE_DF.iloc[:3]
    test_df = SAMPLE_DF.iloc[3:]
    data_prep.save_processed_data(train_df, test_df)

    mock_makedirs.assert_called_with(
        os.path.join("data", "processed", "20230101_000000"),
        exist_ok=True
    )
    assert mock_to_csv.call_count == 2

@patch.object(DataPreparation, 'save_processed_data')
@patch.object(DataPreparation, 'split_data', return_value=(SAMPLE_DF.iloc[:3], SAMPLE_DF.iloc[3:]))
@patch.object(DataPreparation, 'handle_outliers', return_value=SAMPLE_DF)
@patch.object(DataPreparation, 'handle_missing_values', return_value=SAMPLE_DF)
@patch.object(DataPreparation, 'load_data', return_value=SAMPLE_DF)
def test_run(mock_load_data, mock_handle_missing_values, mock_handle_outliers, mock_split_data, mock_save_processed_data, data_prep):
    train_df, test_df = data_prep.run()
    assert len(train_df) == 3
    assert len(test_df) == 2
    mock_load_data.assert_called_once()
    mock_handle_missing_values.assert_called_once()
    mock_handle_outliers.assert_called_once_with(SAMPLE_DF, ['num1', 'num2'])
    mock_split_data.assert_called_once()
    mock_save_processed_data.assert_called_once()