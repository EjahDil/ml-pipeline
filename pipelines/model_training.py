import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import yaml
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ModelTraining:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.model_params = self.config['model_params']
        self.mlflow_config = self.config['mlflow']
        
        mlflow.set_tracking_uri(os.getenv('REGISTRY_URI')) 
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
    
    def get_model(self, model_type):       
        logger.info(f"Creating {model_type} model")

        if model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params['random_forest'])
        elif model_type == 'xgboost':
            return XGBClassifier(**self.model_params['xgboost'])
        elif model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):     
        best_model = None
        best_score = -np.inf
        best_model_type = None
      
        for model_type in self.training_config['model_types']:
            logger.info(f"Training model: {model_type}")

            with mlflow.start_run(run_name=f"{self.mlflow_config['run_name']}_{model_type}"):

                mlflow.log_param("model_type", model_type)
                mlflow.log_params(self.training_config)
                mlflow.log_params(self.model_params[model_type])

                model = self.get_model(model_type)
                model.fit(X_train, y_train)

                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
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

                mlflow.sklearn.log_model(model, name=f"model_{model_type}")


                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_type = model_type

                logger.info(f"Best model: {best_model_type} with score {best_score:.4f}")
                return best_model, best_model_type
