import yaml
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
import joblib
import logging
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ColumnEncoding:
    name: str
    strategy: str
    mapping: Dict[str, Any] = None
    categories: List[str] = None

class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    def __init__(self, config: Union[str, Path, Dict], unknown_token: str = "<UNK>", apply_selection: bool = True):
        self.unknown_token = unknown_token
        self.columns: List[str] = []
        self.encodings: Dict[str, ColumnEncoding] = {}
        self.encoders: Dict[str, Any] = {}
        self._config_hash = None
        self._fitted = False
        self.scaler = None
        self.apply_selection = apply_selection
        
        if isinstance(config, (str, Path)):
            config = Path(config)
            with open(config) as f:
                config_data = yaml.safe_load(f) or {}
            self._config_hash = self._hash_file(config)
        else:
            config_data = config
            self._config_hash = self._hash_dict(config_data)
        
        self.selected_features = None
        self._parse_config(config_data)
        self.scaler_type = config_data.get("preprocessing", {}).get("scaler", "standard")  # Default to standard
        self.feature_creations = config_data.get("feature_creations", [])  # List of dicts for custom features, e.g., [{"name": "TotalCalls", "formula": "InboundCalls + OutboundCalls"}]
    
    def _hash_file(self, path: Path) -> str:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _hash_dict(self, d: dict) -> str:
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()
    
    def _parse_config(self, config: Dict):
        encodings = config.get("encoding", {})
        features = config.get("features", {})
        self.categorical = features.get("categorical", [])
        self.numerical = features.get("numerical", [])
        self.selected_features = features.get("selected_features", self.numerical + self.categorical)
        
        for col, spec in encodings.items():           
            strategy = spec.get("strategy", "mapping").lower()  # Renamed "binary" to "mapping" for generality
            if strategy == "mapping":                
                mapping = {str(k).strip(): v for k, v in spec.items() if k not in ["strategy"]}  # Allow any type, not just int
                enc = ColumnEncoding(name=col, strategy="mapping", mapping=mapping)
            elif strategy in ["ordinal", "onehot"]:
                categories = spec.get("categories")
                if not categories:
                    raise ValueError(f"Missing 'categories' for {col}")
                enc = ColumnEncoding(name=col, strategy=strategy, categories=categories)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            if col in self.encodings:
                raise ValueError(f"Duplicate column encoding: {col}")
            self.encodings[col] = enc
            self.columns.append(col)
    
    def drop_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features:
            available_cols = [col for col in self.selected_features if col in X.columns]
            X = X[available_cols]
        return X
    
    def create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'InboundCalls' in X.columns and 'OutboundCalls' in X.columns and 'TotalCalls' not in X.columns:
            X['TotalCalls'] = X['InboundCalls'] + X['OutboundCalls']
            if 'TotalCalls' not in self.numerical:
                self.numerical.append('TotalCalls')
            if self.apply_selection and 'TotalCalls' not in self.selected_features:
                self.selected_features.append('TotalCalls')
        
        return X
    
    def scale_numerical(self, X: pd.DataFrame, is_fit: bool = True) -> pd.DataFrame:
        if not self.numerical:
            return X
        logger.info(f"Scaling numerical features using {self.scaler_type}")
        
        if is_fit:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler: {self.scaler_type}")
            X[self.numerical] = self.scaler.fit_transform(X[self.numerical])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            X[self.numerical] = self.scaler.transform(X[self.numerical])
        
        return X
    