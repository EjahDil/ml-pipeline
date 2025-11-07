import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_main_model_fails_thresholds(mocker, temp_config_file, caplog, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "fake-account")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", "fake-container")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "fake-key-123")
    monkeypatch.setenv("REGISTRY_URI", "file://./mlruns")

    dp_inst = mocker.Mock()
    fe_inst = mocker.Mock()
    train_inst = mocker.Mock()
    eval_inst = mocker.Mock()

    dp_inst.config = {'features': {'target': 'Churn'}}

    mocker.patch("pipelines.pre_processing.DataPreparation", return_value=dp_inst)
    mocker.patch("pipelines.feature_engineering.FeatureEngineering", return_value=fe_inst)
    mocker.patch("pipelines.model_training.ModelTraining", return_value=train_inst)
    mocker.patch("pipelines.model_evaluation.ModelEvaluation", return_value=eval_inst)

    mocker.patch("mlflow.set_tracking_uri")
    fake_run = mocker.Mock()
    fake_run.info.run_id = "fake-run-id-123"
    mocker.patch("mlflow.start_run", return_value=fake_run)
    mocker.patch("mlflow.log_params")
    mocker.patch("mlflow.log_metrics")
    mocker.patch("mlflow.log_artifact")
    mocker.patch("mlflow.sklearn.log_model")
    mocker.patch("joblib.dump", return_value=None)

    train_df = pd.DataFrame({"Churn": ["Yes", "No"], "f1": [1, 2], "f2": [3, 4]})
    test_df = pd.DataFrame({"Churn": ["No"], "f1": [5], "f2": [6]})

    dp_inst.run.return_value = (train_df, test_df)
    fe_inst.fit.return_value = None
    fe_inst.transform.side_effect = lambda X: X

    train_inst.sanitize_columns.side_effect = lambda cols: list(cols)

    model_mock = mocker.Mock()
    model_mock.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
    train_inst.train.return_value = (model_mock, "random_forest")


    def raise_threshold_error(*args, **kwargs):
        raise ValueError("Model performance does not meet thresholds")

    eval_inst.evaluate.side_effect = raise_threshold_error

    from scripts.train import main
    with pytest.raises(ValueError, match="Model performance"):
        main(temp_config_file)