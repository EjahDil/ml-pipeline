import pytest
import tempfile
from pathlib import Path

FULL_CONFIG_YAML = """
data:
  raw_data: ${RAW_DATA}
  processed_data_path: ${PROCESSED_DATA}
  test_size: 0.2
  random_state: 42

features:
  numerical:
    - MonthlyRevenue
    - MonthlyMinutes
    # ... (all your numerical features)
  categorical:
    - contract_type
    - payment_method
    # ... (all your categorical features)
  target: Churn
  selected_features: ['MonthlyRevenue', 'MonthlyMinutes', 'OverageMinutes', 'UnansweredCalls',
       'CustomerCareCalls', 'PercChangeMinutes', 'PercChangeRevenues',
       'ReceivedCalls', 'TotalRecurringCharge', 'CurrentEquipmentDays',
       'DroppedBlockedCalls', 'MonthsInService', 'ActiveSubs',
       'RespondsToMailOffers', 'RetentionCalls', 'RetentionOffersAccepted',
       'MadeCallToRetentionTeam', 'ReferralsMadeBySubscriber', 'CreditRating',
       'IncomeGroup', 'Occupation', 'PrizmCode','Churn']

preprocessing:
  scaler: standard
  handle_missing: median
  handle_outliers: true
  outlier_method: iqr

encoding:
  MadeCallToRetentionTeam:
    strategy: mapping
    "Yes": 1
    "No": 0
  # ... (rest of encoding)

training:
  model_types: ["random_forest", "xgboost"]
  cv_folds: 5
  scoring: roc_auc
  n_jobs: -1

model_params:
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    random_state: 42
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42

mlflow:
    tracking_uri: "http://mlflow-server:5000"
    experiment_name: "model_comparison_v1"
    run_name: "model_comparison"

monitoring:
  log_metrics: true
  log_params: true
  log_model: true
  log_artifacts: true

azure_blob:
  account_url: "https://azure_account.blob.core.windows.net/mlflow"
  account_key: "${AZURE_STORAGE_ACCOUNT_KEY}"
  container_name: "mlflow"
"""

@pytest.fixture
def temp_config_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(FULL_CONFIG_YAML.strip() + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)