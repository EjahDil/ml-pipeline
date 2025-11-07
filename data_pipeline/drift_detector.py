import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DriftDetector:    
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.reference_data_path = config.get('reference_data_path')
        self.metrics = config.get('metrics', ['psi', 'ks', 'wasserstein'])
        self.thresholds = config.get('thresholds', {})
        self.categorical_columns = config.get('categorical_columns', [])
        self.numerical_columns = config.get('numerical_columns', [])
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        return numerical, categorical
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
       
        try:
            
            if reference.dtype == 'object' or current.dtype == 'object':
                reference_counts = reference.value_counts(normalize=True)
                current_counts = current.value_counts(normalize=True)
                
                
                all_categories = reference_counts.index.union(current_counts.index)
                reference_dist = reference_counts.reindex(all_categories, fill_value=0.001)
                current_dist = current_counts.reindex(all_categories, fill_value=0.001)
            else:
                
                min_val = min(reference.min(), current.min())
                max_val = max(reference.max(), current.max())
                bin_edges = np.linspace(min_val, max_val, bins + 1)
                
                reference_counts, _ = np.histogram(reference.dropna(), bins=bin_edges)
                current_counts, _ = np.histogram(current.dropna(), bins=bin_edges)
                
                reference_dist = (reference_counts + 0.001) / (reference_counts.sum() + 0.001 * bins)
                current_dist = (current_counts + 0.001) / (current_counts.sum() + 0.001 * bins)
                        
            psi = np.sum((current_dist - reference_dist) * np.log(current_dist / reference_dist))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return np.nan
    
    def _calculate_ks_statistic(self, reference: pd.Series, current: pd.Series) -> float:
       
        try:

            if reference.dtype == 'object' or current.dtype == 'object':
                return np.nan
            
            reference_clean = reference.dropna()
            current_clean = current.dropna()
            
            if len(reference_clean) == 0 or len(current_clean) == 0:
                return np.nan
            
            statistic, _ = stats.ks_2samp(reference_clean, current_clean)
            return float(statistic)
            
        except Exception as e:
            logger.error(f"Error calculating KS statistic: {e}")
            return np.nan
    
    def _calculate_wasserstein_distance(self, reference: pd.Series, current: pd.Series) -> float:
       
        try:

            if reference.dtype == 'object' or current.dtype == 'object':
                return np.nan
            
            reference_clean = reference.dropna()
            current_clean = current.dropna()
            
            if len(reference_clean) == 0 or len(current_clean) == 0:
                return np.nan
            
            distance = stats.wasserstein_distance(reference_clean, current_clean)
            
            
            data_range = max(reference_clean.max(), current_clean.max()) - min(reference_clean.min(), current_clean.min())
            if data_range > 0:
                distance = distance / data_range
            
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance: {e}")
            return np.nan
    
    def _calculate_chi_square(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:

        try:

            reference_counts = reference.value_counts()
            current_counts = current.value_counts()
                        
            all_categories = reference_counts.index.union(current_counts.index)
            reference_aligned = reference_counts.reindex(all_categories, fill_value=0)
            current_aligned = current_counts.reindex(all_categories, fill_value=0)
            
            contingency_table = pd.DataFrame({
                'reference': reference_aligned,
                'current': current_aligned
            })
            
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table.T)
            
            return float(chi2), float(p_value)
            
        except Exception as e:
            logger.error(f"Error calculating chi-square: {e}")
            return np.nan, np.nan
    
    async def detect_drift(self, current_data_path: str, reference_data_path: str = None) -> Dict[str, Any]:
       
        if not self.enabled:
            logger.info("Drift detection is disabled")
            return {'enabled': False, 'drift_detected': False}
        
        result = {
            'enabled': True,
            'timestamp': datetime.now().isoformat(),
            'current_data': current_data_path,
            'reference_data': reference_data_path or self.reference_data_path,
            'drift_detected': False,
            'metrics': {},
            'alerts': []
        }
        
        try:

            logger.info(f"Loading current data from: {current_data_path}")
            print("current_data_path")
            print(current_data_path)
            print(current_data_path)
            print(current_data_path)
            print("end_data_path")
            current_df = self._load_data(current_data_path)
            
            ref_path = reference_data_path or self.reference_data_path
            if not ref_path or not Path(ref_path).exists():
                logger.warning(f"Reference data not found at {ref_path}. Saving current data as reference.")
                
                Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
                if Path(current_data_path).suffix == '.parquet':
                    current_df.to_parquet(ref_path)
                else:
                    current_df.to_parquet(ref_path)
                
                return {
                    **result,
                    'message': 'Reference data created from current data',
                    'drift_detected': False
                }
            
            logger.info(f"Loading reference data from: {ref_path}")
            reference_df = self._load_data(ref_path)
            
           
            if not self.numerical_columns and not self.categorical_columns:
                numerical_cols, categorical_cols = self._detect_column_types(reference_df)
            else:
                numerical_cols = self.numerical_columns
                categorical_cols = self.categorical_columns
            
            logger.info(f"Analyzing {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns")
            
            
            for col in reference_df.columns:
                if col not in current_df.columns:
                    logger.warning(f"Column {col} missing in current data")
                    continue
                
                col_metrics = {
                    'column': col,
                    'type': 'numerical' if col in numerical_cols else 'categorical'
                }
                
                if col in numerical_cols:
                    if 'psi' in self.metrics:
                        psi = self._calculate_psi(reference_df[col], current_df[col])
                        col_metrics['psi'] = psi
                        
                        threshold = self.thresholds.get('psi', 0.2)
                        if not np.isnan(psi) and psi >= threshold:
                            result['drift_detected'] = True
                            result['alerts'].append(f"{col}: PSI={psi:.4f} exceeds threshold {threshold}")
                    
                    if 'ks' in self.metrics:
                        ks = self._calculate_ks_statistic(reference_df[col], current_df[col])
                        col_metrics['ks_statistic'] = ks
                        
                        threshold = self.thresholds.get('ks', 0.1)
                        if not np.isnan(ks) and ks >= threshold:
                            result['drift_detected'] = True
                            result['alerts'].append(f"{col}: KS={ks:.4f} exceeds threshold {threshold}")
                    
                    if 'wasserstein' in self.metrics:
                        wasserstein = self._calculate_wasserstein_distance(reference_df[col], current_df[col])
                        col_metrics['wasserstein_distance'] = wasserstein
                        
                        threshold = self.thresholds.get('wasserstein', 0.15)
                        if not np.isnan(wasserstein) and wasserstein >= threshold:
                            result['drift_detected'] = True
                            result['alerts'].append(f"{col}: Wasserstein={wasserstein:.4f} exceeds threshold {threshold}")
                
                elif col in categorical_cols:
                    if 'psi' in self.metrics:
                        psi = self._calculate_psi(reference_df[col], current_df[col])
                        col_metrics['psi'] = psi
                        
                        threshold = self.thresholds.get('psi', 0.2)
                        if not np.isnan(psi) and psi >= threshold:
                            result['drift_detected'] = True
                            result['alerts'].append(f"{col}: PSI={psi:.4f} exceeds threshold {threshold}")
                    
                    chi2, p_value = self._calculate_chi_square(reference_df[col], current_df[col])
                    col_metrics['chi_square'] = chi2
                    col_metrics['chi_square_p_value'] = p_value
                    
                    if not np.isnan(p_value) and p_value < 0.05:
                        result['drift_detected'] = True
                        result['alerts'].append(f"{col}: Chi-square p-value={p_value:.4f} indicates drift")
                
                result['metrics'][col] = col_metrics
            
            result['summary'] = {
                'total_columns_analyzed': len(result['metrics']),
                'columns_with_drift': len(result['alerts']),
                'drift_percentage': len(result['alerts']) / len(result['metrics']) * 100 if result['metrics'] else 0
            }
            
            logger.info(f"Drift detection complete. Drift detected: {result['drift_detected']}")
            if result['alerts']:
                for alert in result['alerts']:
                    logger.warning(f"Drift alert: {alert}")
            
        except Exception as e:
            logger.error(f"Error during drift detection: {e}", exc_info=True)
            result['error'] = str(e)
            result['drift_detected'] = False
        
        return result