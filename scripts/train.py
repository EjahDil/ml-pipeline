import sys
from pathlib import Path
import logging
import argparse
import joblib
import os

from pipelines.pre_processing import DataPreparation
from pipelines.feature_engineering import FeatureEngineering
from pipelines.model_training import ModelTraining
from pipelines.model_evaluation import ModelEvaluation
from dotenv import load_dotenv
import mlflow
import yaml
from datetime import datetime
import tempfile

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    logger.info("Starting ML Training Pipeline")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        os.environ['AZURE_STORAGE_ACCOUNT_NAME'] = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
        os.environ['AZURE_STORAGE_ACCOUNT_KEY'] = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
       
        tracking_uri = os.getenv('REGISTRY_URI')
        mlflow.set_tracking_uri(tracking_uri)

    try:
        logger.info("\nData Preparation")
        
        data_prep = DataPreparation(config_path)
        train_df, test_df = data_prep.run()

        logger.info("\nFeature Engineering")
        
        feature_eng = FeatureEngineering(config_path)
        target_col = data_prep.config['features']['target']


        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        def encode_target(y):
            y_clean = y.astype(str).str.strip().str.capitalize()
            y_encoded = y_clean.map({'No': 0, 'Yes': 1})
            if y_encoded.isna().any():
                raise ValueError(f"Invalid labels: {y_clean[y_encoded.isna()].unique()}")
            return y_encoded.astype(int)
        
        y_train_encoded = encode_target(y_train)
        y_test_encoded = encode_target(y_test)

        
        logger.info(f"Original X_train columns: {list(X_train.columns)}")
        logger.info(f"Original X_test columns: {list(X_test.columns)}")
        
        
        feature_eng.fit(X_train)
        
        
        X_train_transformed = feature_eng.transform(X_train)
        X_test_transformed = feature_eng.transform(X_test)
        
        logger.info(f"Transformed train shape: {X_train_transformed.shape}")
        logger.info(f"Transformed train columns: {list(X_train_transformed.columns)}")
        logger.info(f"Transformed test shape: {X_test_transformed.shape}")
        logger.info(f"Transformed test columns: {list(X_test_transformed.columns)}")
        
        
        if list(X_train_transformed.columns) != list(X_test_transformed.columns):
            logger.error("COLUMN MISMATCH DETECTED!")
            logger.error(f"Train columns: {set(X_train_transformed.columns)}")
            logger.error(f"Test columns: {set(X_test_transformed.columns)}")
            logger.error(f"Only in train: {set(X_train_transformed.columns) - set(X_test_transformed.columns)}")
            logger.error(f"Only in test: {set(X_test_transformed.columns) - set(X_train_transformed.columns)}")
            raise ValueError("Column mismatch between train and test sets")
        
        
        artifact_dir = Path(tempfile.mkdtemp(prefix="mlflow-artifacts-"))
        fe_path = artifact_dir / "feature_engineer.joblib"
        joblib.dump(feature_eng, fe_path)
        feature_eng.save(Path("artifact"))
        mlflow.log_artifact("artifact/feature_engineer.joblib", artifact_path="mlflow-artifacts")
        
        
        model_trainer = ModelTraining(config_path)
        
        
        X_train_transformed.columns = model_trainer.sanitize_columns(X_train_transformed.columns)
        X_test_transformed.columns = model_trainer.sanitize_columns(X_test_transformed.columns)
        
        logger.info(f"Sanitized train columns: {list(X_train_transformed.columns)}")
        logger.info(f"Sanitized test columns: {list(X_test_transformed.columns)}")
        
        
        model, model_type = model_trainer.train(X_train_transformed, y_train_encoded)
        
        logger.info(f"\nModel Evaluation for {model_type}")
        
        
        evaluator = ModelEvaluation(config_path)
        metrics = evaluator.evaluate(model, X_test_transformed, y_test_encoded, feature_eng)
        meets_thresholds = evaluator.check_thresholds(metrics)

        
        predictions = model.predict(X_train_transformed)
        probabilities = model.predict_proba(X_train_transformed)[:, 1]
        
        result = X_train.copy()
        result["prediction"] = predictions
        result["probability"] = probabilities
        now = datetime.now()
        result.to_csv(f"data/predictions_{now}.csv", index=False)
        mlflow.log_artifact(f"data/predictions_{now}.csv", artifact_path="mlflow-artifacts")
        logger.info("Done! Predictions saved.")        

        logger.info("Cleaning Up Local")
        os.remove(f"data/predictions_{now}.csv")
        # os.remove(f"artifact/feature_engineer.joblib")
        if meets_thresholds:
            logger.info("\nModel meets all performance thresholds!")
        else:
            logger.warning("\nModel does not meet some performance thresholds")
                     
        logger.info("Training Pipeline Completed Successfully!")
       
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yml",
        help="Path to training config file"
    )
    
    args = parser.parse_args()
    main(args.config)