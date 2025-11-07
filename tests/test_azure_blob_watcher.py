import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open
import asyncio

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.azure_watcher import AzureBlobWatcher 

@pytest.fixture
def config():
    return {
        "azure": {
            "container_name": "testcontainer",
            "storage_account_name": "testacc",
            "storage_account_key": "testkey"
        },
        "watcher": {
            "state_file": ".azure_watcher_state.json"
        }
        # Add any other config keys your AzureBlobWatcher expects
    }

@pytest.fixture
def config(tmp_path):
    return {
        "azure": {
            "storage_account_name": "testacc",
            "storage_account_key": "testkey",
            "container_name": "testcontainer"
        },
        "watcher": {
            "state_file": str(tmp_path / ".azure_watcher_state.json"),
            "download_dir": str(tmp_path / "downloads")
        }
    }


def mock_blob(name, size=100, last_modified=None, md5=None):
    blob = MagicMock()
    blob.name = name
    blob.size = size
    blob.last_modified = last_modified or datetime.utcnow()
    blob.content_settings.content_md5 = bytes.fromhex(md5) if md5 else None
    return blob


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testacc")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "testkey")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER_NAME", "testcontainer")

@patch("builtins.open", new_callable=mock_open, read_data="{}")
@patch("pathlib.Path.exists", return_value=True)
@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_no_new_blobs(mock_client_init, mock_open, mock_exists, config):
    container_client = MagicMock()
    container_client.list_blobs.return_value = []
    mock_client_init.return_value.get_container_client.return_value = container_client

    watcher = AzureBlobWatcher(config)
    result = asyncio.run(watcher.check_new_blobs())
    assert result == []


@patch("builtins.open", new_callable=mock_open, read_data=b"content")
@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_upload_file(mock_client_init, mock_open, config, tmp_path):
    blob_client = MagicMock()
    blob_service_client = MagicMock()
    blob_service_client.get_blob_client.return_value = blob_client
    mock_client_init.return_value = blob_service_client

    local_file = tmp_path / "test.txt"
    local_file.write_bytes(b"content")

    watcher = AzureBlobWatcher(config)
    blob_name = asyncio.run(watcher.upload_file(str(local_file)))

    assert blob_name == "test.txt"
    blob_client.upload_blob.assert_called_once()
    handle = mock_open()
    handle.seek.assert_called_with(0)