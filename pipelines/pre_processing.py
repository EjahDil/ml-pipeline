import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import yaml
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.preprocessing_config = self.config['preprocessing']
        self.raw_data = self.data_config['raw_data']
    
    def load_data(self) -> pd.DataFrame:    
        logger.info("Loading data")
        df = pd.read_csv(self.data_config['raw_data'])
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.preprocessing_config['handle_missing']
        logger.info(f"Handling missing values using {method} method")
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    if method == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
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
    
    