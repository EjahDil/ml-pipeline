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
    
    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        X = self.create_features(X)
        if self.apply_selection:
            X = self.drop_features(X)
            self.numerical = [col for col in self.numerical if col in X.columns]
        
        X = self.scale_numerical(X, is_fit=True)
        
        missing_cols = set(self.columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        for col, enc in self.encodings.items():
            col_data = X[col].astype(str)
            if enc.strategy == "mapping":
                valid_keys = set(enc.mapping.keys())
                seen = set(col_data.unique())
                invalid = seen - valid_keys
                if invalid:
                    logger.warning(f"Invalid values in {col}: {list(invalid)[:5]} → mapped to NaN or default")
                mapping = enc.mapping.copy()
                if self.unknown_token not in mapping:
                    mapping[self.unknown_token] = np.nan  # Or configurable default
                self.encoders[col] = mapping
            elif enc.strategy == "ordinal":
                categories = enc.categories[:]
                if self.unknown_token not in categories:
                    categories.append(self.unknown_token)
                    enc.categories = categories
                oe = OrdinalEncoder(
                    categories=[categories],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                oe.fit(X[[col]])
                self.encoders[col] = oe
            elif enc.strategy == "onehot":
                categories = enc.categories[:]
                if self.unknown_token not in categories:
                    categories.append(self.unknown_token)
                    enc.categories = categories
                ohe = OneHotEncoder(
                    categories=[categories],
                    handle_unknown='ignore',
                    sparse_output=False
                )
                ohe.fit(X[[col]])
                self.encoders[col] = ohe
        
        self._fitted = True
        logger.info(f"Fitted on {len(self.columns)} categorical and {len(self.numerical)} numerical columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Must call fit() first")
        
        X = pd.DataFrame(X).copy()
        X = self.create_features(X)
        if self.apply_selection:
            X = self.drop_features(X)
            self.numerical = [col for col in self.numerical if col in X.columns]
        
        X = self.scale_numerical(X, is_fit=False)
        
        results = []
        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Missing column: {col}")
            
            enc = self.encodings[col]
            col_data = X[col].astype(str)
            
            if enc.strategy == "mapping":
                mapped = col_data.map(self.encoders[col]).fillna(np.nan)
                results.append(pd.DataFrame(mapped.values, columns=[col], index=X.index))
            elif enc.strategy == "ordinal":
                transformed = self.encoders[col].transform(col_data.to_frame())
                results.append(pd.DataFrame(transformed, columns=[col], index=X.index))
            elif enc.strategy == "onehot":
                transformed = self.encoders[col].transform(col_data.to_frame())
                ohe_cols = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                results.append(pd.DataFrame(transformed, columns=ohe_cols, index=X.index))
        
        encoded_df = pd.concat(results, axis=1)

        full_df = pd.concat([X[self.numerical], encoded_df], axis=1)
        return full_df
    
    def get_feature_names_out(self) -> List[str]:
        names = self.numerical.copy()
        for col in self.columns:
            enc = self.encodings[col]
            if enc.strategy in ["mapping", "ordinal"]:
                names.append(col)
            elif enc.strategy == "onehot":
                names.extend([f"{col}_{cat}" for cat in enc.categories])
        return names
    
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.get_feature_names_out())
        
        result = X[self.numerical].copy()
        if self.scaler:
            result[self.numerical] = self.scaler.inverse_transform(result[self.numerical])
        
        for col in self.columns:
            enc = self.encodings[col]
            if enc.strategy == "mapping":
                inv_map = {v: k for k, v in self.encoders[col].items() if not pd.isna(v)}
                result[col] = X[col].map(inv_map).fillna(self.unknown_token)
            elif enc.strategy == "ordinal":
                inv = self.encoders[col].inverse_transform(X[[col]])
                result[col] = inv.ravel()
            elif enc.strategy == "onehot":
                ohe_cols = [f"{col}_{cat}" for cat in enc.categories]
                inv = self.encoders[col].inverse_transform(X[ohe_cols])
                result[col] = inv.ravel()
        
        return result
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "feature_engineer.joblib")
        metadata = {
            "config_hash": self._config_hash,
            "columns": self.columns,
            "numerical": self.numerical,
            "feature_names": self.get_feature_names_out()
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Feature engineer saved to {path}")
    
    @classmethod
    def load(cls, path: Path):
        engineer = joblib.load(path / "feature_engineer.joblib")
        logger.info(f"Loaded with {len(engineer.columns)} categorical and {len(engineer.numerical)} numerical columns")
        return engineer