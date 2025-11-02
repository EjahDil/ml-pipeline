import sys
from pathlib import Path
import logging
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.pre_processing import DataPreparation
from pipelines.feature_engineering import FeatureEngineering

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    logger.info("Starting ML Training Pipeline")
    
    try:
        logger.info("\nData Preparation")
        
        data_prep = DataPreparation(config_path)
        train_df, test_df = data_prep.run()

        logger.info("\nFeature Engineering")
        
        feature_eng =  FeatureEngineering(config_path)
        target_col = data_prep.config['features']['target']

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        feature_eng.fit(X_train)

        X_train_transformed = feature_eng.transform(X_train)
        X_test_transformed = feature_eng.transform(X_test)
        
        logger.info("Transformed train shape: %s", X_train_transformed.shape)
        logger.info("Transformed test shape: %s", X_test_transformed.shape)
       
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