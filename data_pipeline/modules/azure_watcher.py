import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobProperties
from azure.core.exceptions import AzureError
import hashlib
import json


from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)


class AzureBlobWatcher:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        self.ref_blob_name = ""
        self.blob_prefix = config.get('blob_prefix', '')
        
        self.state_file = Path('.azure_watcher_state.json')
        self.processed_blobs = self._load_state()
        
        self._init_client()
    
    def _init_client(self):
      
        try:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.storage_account_name};"
                f"AccountKey={self.storage_account_key};"
                f"EndpointSuffix=core.windows.net"
            )
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            logger.info(f"Connected to Azure Blob Storage: {self.container_name}")
           
        except AzureError as e:
            logger.error(f"Failed to connect to Azure Blob Storage: {e}")
            raise
    
    def _load_state(self) -> Dict[str, Dict[str, Any]]:

        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state with {len(state)} processed blobs")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}. Starting fresh.")
                return {}
        return {}
    
    def _save_state(self):

        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.processed_blobs, f, indent=2)
            logger.debug(f"State saved with {len(self.processed_blobs)} blobs")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _get_blob_hash(self, blob_props: BlobProperties) -> str:
       
        if hasattr(blob_props, 'content_settings') and blob_props.content_settings.content_md5:
            return blob_props.content_settings.content_md5.hex()
        
        hash_str = f"{blob_props.name}_{blob_props.last_modified}_{blob_props.size}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    async def check_new_blobs(self) -> List[str]:
        
        new_blobs = []
        try:

            blob_list = self.container_client.list_blobs(name_starts_with=self.blob_prefix)
           
            for blob in blob_list:
               
              
                if blob.size == 0:
                    continue
                
                blob_hash = self._get_blob_hash(blob)
                blob_name = blob.name
                
                if blob_name not in self.processed_blobs:
                    logger.info(f"New blob detected: {blob_name}")
                    new_blobs.append(blob_name)
                    self.processed_blobs[blob_name] = {
                        'hash': blob_hash,
                        'last_modified': blob.last_modified.isoformat(),
                        'size': blob.size,
                        'processed_at': datetime.now().isoformat()
                    }
                elif self.processed_blobs[blob_name]['hash'] != blob_hash:
                    logger.info(f"Modified blob detected: {blob_name}")
                    new_blobs.append(blob_name)
                    self.processed_blobs[blob_name].update({
                        'hash': blob_hash,
                        'last_modified': blob.last_modified.isoformat(),
                        'size': blob.size,
                        'processed_at': datetime.now().isoformat()
                    })
            

            if new_blobs:
                self._save_state()
            print(f"non {new_blobs}")
            return new_blobs
            
        except AzureError as e:
            logger.error(f"Error checking for new blobs: {e}")
            raise
    
    async def download_blob(self, blob_name: str, local_dir: str) -> str:
       
        try:
            local_path = Path(local_dir) / Path(blob_name).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            logger.info(f"Downloading {blob_name} to {local_path}")
            
           
            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            
            logger.info(f"Successfully downloaded {blob_name} ({local_path.stat().st_size} bytes)")
            return str(local_path)
            
        except AzureError as e:
            logger.error(f"Failed to download blob {blob_name}: {e}")
            raise
    async def download_reference(self):

        blob_client =self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob="data/combined.csv"
            )       
        with open("data/ref_data.csv", "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

    
    async def upload_file(self, local_path: str, blob_name: str = None) -> str:
       
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")
    
        try:
            if blob_name is None:
                blob_name = local_path.name  
    
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            logger.info("Uploading %s to container=%s blob=%s", local_path, self.container_name, blob_name)
    

            with open(local_path, "rb") as f:
                f.seek(0) 
                blob_client.upload_blob(f, overwrite=True, timeout=300)
    
            logger.info("Successfully uploaded: %s", blob_name)
            print(f"Upload completed: {blob_name}")
            return blob_name
    
        except AzureError as e:
            logger.error("Azure upload failed for %s: %s", local_path, e)
            raise
        except Exception as e:
            logger.error("Unexpected error during upload: %s", e)
            raise

    async def upload_combined_file(self, local_path: str) -> str:        
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob="data/combined_1.csv" #self.ref_blob_name,
            )

            logger.info("Uploading %s to container=%s blob=%s",
                        local_path, self.container_name, self.ref_blob_name)


            with open(local_path, "rb") as f:
                f.seek(0)
                blob_client.upload_blob(f, overwrite=True, timeout=300)
            logger.info("Upload completed: %s", self.ref_blob_name)
            return self.ref_blob_name

        except ResourceNotFoundError:
            logger.error("Container %s does not exist", self.container_name)
            raise
        except ResourceExistsError:
            logger.warning("Blob already exists – overwriting because overwrite=True")
        except AzureError as e:
            logger.error("Azure error while uploading %s: %s", local_path, e)
            raise
        except Exception as e:
            logger.exception("Unexpected error during upload")
            raise
    
    def cleanup_old_state(self, days: int = 30):
        cutoff_date = datetime.now() - timedelta(days=days)
        
        blobs_to_remove = []
        for blob_name, blob_info in self.processed_blobs.items():
            processed_at = datetime.fromisoformat(blob_info['processed_at'])
            if processed_at < cutoff_date:
                blobs_to_remove.append(blob_name)
        
        for blob_name in blobs_to_remove:
            del self.processed_blobs[blob_name]
        
        if blobs_to_remove:
            logger.info(f"Cleaned up {len(blobs_to_remove)} old blob entries from state")
            self._save_state()