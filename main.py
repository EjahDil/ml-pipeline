import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml
import sys
import pandas as pd
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.modules.azure_watcher import AzureBlobWatcher
from data_pipeline.modules.dvc_manager import DVCManager
from data_pipeline.modules.drift_detector import DriftDetector
from scripts.train import train

load_dotenv()


logger = logging.getLogger(__name__)


class DataPipeline:
    
    def __init__(self, config_path: str = "configs/train_config.yml"):
        self.config = self._load_config(config_path)
        logger.info("config initialized")
        self._validate_config()
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                config_str = f.read()
               
                for key, value in os.environ.items():
                    config_str = config_str.replace(f'${{{key}}}', value)
                config = yaml.safe_load(config_str)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate_config(self):
        logger.info("config validation started")
        required_keys = ['azure', 'dvc', 'drift_detection']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")
        
        
        if not all([
            os.getenv('AZURE_STORAGE_ACCOUNT'),
            os.getenv('AZURE_STORAGE_KEY'),
            os.getenv('AZURE_CONTAINER_NAME')
        ]):
            raise ValueError("Missing required Azure credentials in environment variables")
        
        logger.info("Configuration validation passed")
    
    def _initialize_components(self):

        try:
            
            self.blob_watcher = AzureBlobWatcher(self.config['azure'])

            self.dvc_manager = DVCManager(self.config['dvc'])
            self.drift_detector = DriftDetector(self.config['drift_detection'])
            
            Path("logs").mkdir(exist_ok=True)
            Path(self.config['dvc']['data_dir']).mkdir(parents=True, exist_ok=True)
            Path(self.config['storage']['local_cache_dir']).mkdir(parents=True, exist_ok=True)
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def process_new_data(self, blob_path: str) -> Dict[str, Any]:
        
        pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"[{pipeline_id}] Starting pipeline for blob: {blob_path}")
        
        results = {
            'pipeline_id': pipeline_id,
            'blob_path': blob_path,
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
           
            logger.info(f"[{pipeline_id}] Step 1: Downloading data from Azure Blob")
            
            local_path = await self.blob_watcher.download_blob(
                blob_path,
                self.config['dvc']['data_dir']
            )
           
            results['steps']['download'] = {'status': 'success', 'local_path': local_path}                          

            logger.info(f"[{pipeline_id}] Step 2: Versioning data with DVC")
            dvc_result = await self.dvc_manager.version_data(local_path)
            results['steps']['dvc_version'] = dvc_result

            if not dvc_result['success']:
                error = dvc_result.get('error', '')
                if "nothing to commit" in str(error).lower():
                    logger.info("Data already versioned - continuing pipeline")
                  
                else:
                    raise Exception(f"DVC versioning failed: {error}")
            
           
            await self.blob_watcher.download_reference()            
           
            new_ingested_data = pd.read_csv(local_path)
            reference_data = pd.read_csv("data/ref_data.csv")
            combined = pd.concat([new_ingested_data, reference_data], ignore_index=True)
            azure_blob_watcher = AzureBlobWatcher(self.config['azure'])
            combined.to_csv("data/reference/combined.csv", index=False)            
            
            await azure_blob_watcher.upload_file("data/reference/combined.csv", "data/combined_2.csv")

            if self.config['drift_detection']['enabled']:
                logger.info(f"[{pipeline_id}] Step 3: Performing drift detection")
                drift_result = await self.drift_detector.detect_drift(
                    local_path,
                    "ref_data.csv",                   
                )
               
                results['steps']['drift_detection'] = drift_result

                if drift_result['drift_detected'] and self.config['drift_detection']['trigger_on_drift']:
                    logger.warning(f"[{pipeline_id}] Drift detected! Triggering model training")
                       
                    train("configs/train_config.yml")
                    
                else:
                    logger.info(f"[{pipeline_id}] No significant drift detected. Skipping model training.")
                    results['steps']['model_training'] = {'status': 'skipped', 'reason': 'no_drift'}                    
            
            logger.info(f"[{pipeline_id}] Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"[{pipeline_id}] Pipeline failed: {e}", exc_info=True)
           
            raise
        
        return results
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        
        try:
            results_path = Path(self.config['monitoring']['metrics_path']) / f"pipeline_{results['pipeline_id']}.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            logger.info(f"Pipeline results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")
    
    async def start_scheduler(self):
        
        logger.info("Starting data monitoring scheduler...")
        check_interval = self.config['scheduler']['check_interval_seconds']
       
        while True:
            try:
                new_blobs = await self.blob_watcher.check_new_blobs()
                print(f"checking {new_blobs}")
                if new_blobs:
                    logger.info(f"Found {len(new_blobs)} new blob(s)")
                    print(f"Found {len(new_blobs)} new blob(s)")
                    for blob_path in new_blobs:
                        await self.process_new_data(blob_path)
                else:
                    logger.debug("No new blobs found")
                    print("No new blobs found")
                
                await asyncio.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(check_interval)
    
    async def manual_trigger(self, blob_path: str) -> Dict[str, Any]:

        logger.info(f"Manual trigger requested for blob: {blob_path}")
        return await self.process_new_data(blob_path)



async def main():
    try:
        pipeline = DataPipeline()
        if len(sys.argv) > 1 and sys.argv[1] == "manual":

            if len(sys.argv) < 3:
               
                sys.exit(1)
            blob_path = sys.argv[2]
            result = await pipeline.manual_trigger(blob_path)
            print(f"Pipeline completed: {result}")
        else:
            await pipeline.start_scheduler()
            
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    print("initialized")
    asyncio.run(main())    