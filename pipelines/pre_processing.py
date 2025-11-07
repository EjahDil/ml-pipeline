from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import yaml
import logging
from datetime import datetime, timezone


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.preprocessing_config = self.config['preprocessing']
    
    def load_data(self) -> pd.DataFrame:    
        logger.info("Loading data")

        RAW_DATA_SOURCE = ""
        secret_path = Path("/run/secrets/raw_data_source.txt")
        if secret_path.exists():
            RAW_DATA_SOURCE = secret_path.read_text().strip()
        else:

            RAW_DATA_SOURCE = os.getenv("RAW_DATA_SOURCE")
            
        df = pd.read_csv(RAW_DATA_SOURCE)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.preprocessing_config['handle_missing']
        logger.info(f"Handling missing values using {method} method")
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    if method == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif method == 'median':
                        df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        if not self.preprocessing_config['handle_outliers']:
            return df
        
        logger.info("Handling outliers using IQR method")
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        test_size = self.data_config['test_size']
        random_state = self.data_config['random_state']
        
        logger.info(f"Splitting data with test_size={test_size}")
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[self.config['features']['target']] if self.config['features']['target'] in df.columns else None
        )
        test_df.columns = train_df.columns
        logger.info(f"Train columns: {list(train_df.columns)}")
        logger.info(f"Test columns: {list(test_df.columns)}")

        return train_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):       
        logger.info("Saving processed data")
        
        timestamp = datetime.now(timezone.utc)
        processing_id = timestamp.strftime('%Y%m%d_%H%M%S')
        

        output_dir = os.path.join("data", "processed", processing_id)
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train.csv")
        train_df.to_csv(train_path, index=False)
        logger.info(f"Train data saved to {train_path} ({len(train_df)} records)")

        test_path = os.path.join(output_dir, "test.csv")
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test data saved ({len(test_df)} records)")        
    
    def run(self):
        df = self.load_data()
        
        df = self.handle_missing_values(df)
        
        numerical_cols = self.config['features']['numerical']
        df = self.handle_outliers(df, numerical_cols)
        
        train_df, test_df = self.split_data(df)
        
        self.save_processed_data(train_df, test_df)
        
        logger.info("Data preparation completed successfully")
        return train_df, test_df