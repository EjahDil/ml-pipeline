import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging

from .feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)


class ModelEvaluation:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.thresholds = self.config.get('performance_thresholds', {})
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, feature_eng: FeatureEngineering) -> dict:
     
        logger.info(f"Engineered test shape: {X_test.shape}")

        logger.info(f"Feature names: {list(X_test.columns)}")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")

        report = classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")

        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        self.plot_confusion_matrix(y_test, y_pred)

        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, X_test.columns)

        mlflow.log_param("num_features_engineered", len(X_test.columns))
        mlflow.log_param("selected_features_count", len(feature_eng.selected_features))

        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
    
    def plot_feature_importance(self, model, feature_names):

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] 
        
        plt.figure(figsize=(10, 8))
        plt.title("Top 20 Feature Importances")
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
    
    def check_thresholds(self, metrics: dict) -> bool:
        
        logger.info("Checking performance thresholds...")
        
        meets_thresholds = True
        for metric_name, threshold in self.thresholds.items():
            metric_key = f"min_{metric_name}"
            if metric_key in self.thresholds:
                actual_value = metrics.get(metric_name, 0)
                threshold_value = self.thresholds[metric_key]
                
                if actual_value < threshold_value:
                    logger.warning(
                        f"{metric_name} ({actual_value:.4f}) is below threshold ({threshold_value:.4f})"
                    )
                    meets_thresholds = False
                else:
                    logger.info(
                        f"{metric_name} ({actual_value:.4f}) meets threshold ({threshold_value:.4f})"
                    )
        
        return meets_thresholds