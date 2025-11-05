

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import yaml
import logging
import os
from dotenv import load_dotenv

import tempfile
import shutil


load_dotenv()

logger = logging.getLogger(__name__)


class ModelTraining:
    
    def __init__(self, config_path: str):
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
        self.training_config = self.config['training']
        self.model_params = self.config['model_params']
        self.mlflow_config = self.config['mlflow']
        mlflow.set_experiment(
            experiment_name= self.mlflow_config['experiment_name']
        )
        artifact_uri = (
            f"wasbs://{os.getenv('AZURE_STORAGE_CONTAINER_NAME')}"
            f"@{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net/mlflow"
        )
        mlflow_client = mlflow.tracking.MlflowClient()
        exp = mlflow_client.get_experiment_by_name(self.mlflow_config['experiment_name'])
        if exp is None:
            exp_id = mlflow_client.create_experiment(
                name= self.mlflow_config['experiment_name'],
                artifact_location=artifact_uri
            )
        else:
            exp_id = exp.experiment_id

        experiment_id = exp_id
    
    def get_model(self, model_type):       
        logger.info(f"Creating {model_type} model")

        if model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params['random_forest'])
        elif model_type == 'xgboost':
            return XGBClassifier(**self.model_params['xgboost'])       
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def upload_to_azure_blob(self, local_path: str, remote_path: str):
        
        try:
            container_client = self.blob_service_client.get_container_client(
                self.azure_config['container_name']
            )
            
            if os.path.isfile(local_path):
                
                blob_client = container_client.get_blob_client(remote_path)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.info(f"Uploaded {local_path} to {remote_path}")
            
            elif os.path.isdir(local_path):
                
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, local_path)
                        blob_path = f"{remote_path}/{relative_path}"
                        
                        blob_client = container_client.get_blob_client(blob_path)
                        with open(file_path, "rb") as data:
                            blob_client.upload_blob(data, overwrite=True)
                        logger.info(f"Uploaded {file_path} to {blob_path}")
            
            logger.info(f"Successfully uploaded artifacts to Azure Blob: {remote_path}")
            
        except Exception as e:
            logger.error(f"Error uploading to Azure Blob: {str(e)}")
            raise
   
    def sanitize_columns(self, columns):
        return (
            columns.str.strip()
            .str.lower()
            .str.replace(r'[^\w]+', '_', regex=True) 
            .str.replace(r'_+', '_', regex=True)
            .str.strip('_')
            .str.replace('unk', '__UNK__', case=False) 
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):     
        best_model = None
        self.best_score = -np.inf
        best_model_type = None
    
        for model_type in self.training_config['model_types']:
            logger.info(f"Training model: {model_type}")

            with mlflow.start_run(run_name=f"{self.mlflow_config['run_name']}_{model_type}", nested=True):

                mlflow.log_param("model_type", model_type)
                mlflow.log_params(self.training_config)
                mlflow.log_params(self.model_params[model_type])

                model = self.get_model(model_type)
              
                unique_labels = y_train.unique()
                if not set(unique_labels).issubset({0, 1}):
                    logger.warning(f"Expected y_train to be encoded as 0/1, got: {unique_labels}")
                    
                    y_clean = y_train.astype(str).str.strip().str.capitalize()
                    y_encoded = y_clean.map({'No': 0, 'Yes': 1})
                    if y_encoded.isna().any():
                        raise ValueError(f"Invalid labels: {y_clean[y_encoded.isna()].unique()}")
                    y_encoded = y_encoded.astype(int)
                else:
                    y_encoded = y_train.astype(int)

                X_train_transformed = X_train.copy()
                X_train.columns = self.sanitize_columns(X_train.columns)
                logger.info(f"Using {X_train_transformed.shape[1]} sanitized features")
                model.fit(X_train, y_encoded)

                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_encoded,
                    cv=self.training_config['cv_folds'],
                    scoring=self.training_config['scoring'],
                    n_jobs=self.training_config['n_jobs']
                )

                mean_score = cv_scores.mean()
                std_score = cv_scores.std()

                mlflow.log_metric("cv_mean_score", mean_score)
                mlflow.log_metric("cv_std_score", std_score)

                logger.info(f"{model_type} {self.training_config['scoring']}: "
                            f"{mean_score:.4f} (+/- {std_score:.4f})")

                predictions = model.predict(X_train)
                probabilities = model.predict_proba(X_train)[:, 1]

                signature = infer_signature(X_train, predictions)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"model_{model_type}",
                    signature=signature,
                )

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    best_model = model
                    best_model_type = model_type

                logger.info(f"Current best model: {best_model_type} with score {self.best_score:.4f}")
                mlflow.end_run()

        if best_model is not None:
            logger.info(f"Saving best model: {best_model_type} with score {self.best_score:.4f}")

            with mlflow.start_run(run_name=f"best_model_{best_model_type}", nested=True):
                mlflow.log_param("best_model_type", best_model_type)
                mlflow.log_metric("best_cv_score", self.best_score)

                predictions = best_model.predict(X_train)
                signature = infer_signature(X_train, predictions)

                mlflow.sklearn.log_model(
                    best_model,
                    "best_model",
                    registered_model_name=f"{model_type}_model",
                    signature=signature,
                    input_example=X_train.iloc[:1]
                )

                run_id = mlflow.active_run().info.run_id
                logger.info(f"Best model run ID: {run_id}")

        return best_model, best_model_type

    def save_all_artifacts_to_azure(self, X_train: pd.DataFrame, y_train: pd.Series):
        
        best_model, best_model_type = self.train(X_train, y_train)
        
        if best_model is None:
            raise ValueError("No models were trained successfully")
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.mlflow_config['experiment_name']
        
        temp_dir = tempfile.mkdtemp(prefix="mlflow_artifacts_")
        
        try:
            logger.info("Downloading MLflow artifacts...")
            artifacts_uri = f"mlruns/{experiment_name}"
            
            mlflow.artifacts.download_artifacts(
                artifact_uri=artifacts_uri,
                dst_path=temp_dir
            )
            
            blob_path = f"models/{experiment_name}/{best_model_type}_{timestamp}"
            
            logger.info(f"Uploading all artifacts to Azure Blob: {blob_path}")
            self.upload_to_azure_blob(temp_dir, blob_path)
            
            import pickle
            model_pickle_path = os.path.join(temp_dir, "best_model.pkl")
            with open(model_pickle_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            self.upload_to_azure_blob(model_pickle_path, f"{blob_path}/best_model.pkl")
            
            logger.info(f"   All artifacts saved to Azure Blob:")
            logger.info(f"   Blob Path: {blob_path}")
            logger.info(f"   Best Model: {best_model_type}")
            logger.info(f"   Score: {self.best_score:.4f}")
            
            return {
                "blob_path": blob_path,
                "best_model_type": best_model_type,
                "best_score": self.best_score,
                "timestamp": timestamp
            }
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)