#!/bin/bash
set -e

if [ -f /run/secrets/azure_storage_connection_string.txt ]; then
  export AZURE_STORAGE_CONNECTION_STRING=$(cat /run/secrets/azure_storage_connection_string.txt)
fi

if [ -f /run/secrets/azure_storage_account_name.txt ]; then
  export AZURE_STORAGE_ACCOUNT_NAME=$(cat /run/secrets/azure_storage_account_name.txt)
fi

if [ -f /run/secrets/azure_storage_account_key.txt ]; then
  export AZURE_STORAGE_ACCOUNT_KEY=$(cat /run/secrets/azure_storage_account_key.txt)
fi

echo "Loaded secrets:"
[ -n "$AZURE_STORAGE_ACCOUNT_NAME" ] && echo "- AZURE_STORAGE_ACCOUNT_NAME loaded"
[ -n "$AZURE_STORAGE_CONNECTION_STRING" ] && echo "- AZURE_STORAGE_CONNECTION_STRING loaded"
[ -n "$AZURE_STORAGE_ACCOUNT_KEY" ] && echo "- AZURE_STORAGE_ACCOUNT_KEY loaded"

exec "$@"
