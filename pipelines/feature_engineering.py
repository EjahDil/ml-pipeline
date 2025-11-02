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
    
    def __init__(self, config: Union[str, Path, Dict], unknown_token: str = "<UNK>", apply_selection: bool = True, missing_strategy: str = "fill_unknown"):
        self.unknown_token = unknown_token
        self.columns: List[str] = []
        self.encodings: Dict[str, ColumnEncoding] = {}
        self.encoders: Dict[str, Any] = {}
        self._config_hash = None
        self._fitted = False
        self.scaler = None
        self.apply_selection = apply_selection
        self.missing_strategy = missing_strategy 
        
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
        self.scaler_type = config_data.get("preprocessing", {}).get("scaler", "standard") 
        self.feature_creations = config_data.get("feature_creations", []) 
    
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
            strategy = spec.get("strategy", "mapping").lower() 
            if strategy == "mapping":                
                mapping = {str(k).strip(): v for k, v in spec.items() if k not in ["strategy"]} 
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
    
    