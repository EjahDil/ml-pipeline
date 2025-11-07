#!/bin/bash

set -e

if [ -f /run/secrets/azure_storage_connection_string ]; then
  export AZURE_STORAGE_CONNECTION_STRING=$(cat /run/secrets/azure_storage_connection_string)
fi

if [ -f /run/secrets/azure_storage_account_name ]; then
  export AZURE_STORAGE_ACCOUNT_NAME=$(cat /run/secrets/azure_storage_account_name)
fi

if [ -f /run/secrets/azure_storage_account_key ]; then
  export AZURE_STORAGE_ACCOUNT_KEY=$(cat /run/secrets/azure_storage_account_key)
fi

if [ -f /run/secrets/azure_storage_container_name ]; then
  export AZURE_STORAGE_CONTAINER_NAME=$(cat /run/secrets/azure_storage_container_name)
fi

if [ -f /run/secrets/git_repo_url ]; then
  export GIT_REPO_URL=$(cat /run/secrets/git_repo_url)
fi

if [ -f /run/secrets/git_user_email ]; then
  export GIT_USER_EMAIL=$(cat /run/secrets/git_user_email)
fi

if [ -f /run/secrets/git_user_name ]; then
  export GIT_USER_NAME=$(cat /run/secrets/git_user_name)
fi

if [ -f /run/secrets/git_remote ]; then
  export GIT_REMOTE=$(cat /run/secrets/git_remote)
fi

if [ -f /run/secrets/git_branch ]; then
  export GIT_BRANCH=$(cat /run/secrets/git_branch)
fi

if [ -f /run/secrets/azure_dvc_container ]; then
  export AZURE_DVC_CONTAINER=$(cat /run/secrets/azure_dvc_container)
fi

touch data/.gitignore
touch .gitignore
mkdir -p data/reference

git config --global user.name $GIT_USER_NAME
git config --global user.email $GIT_USER_EMAIL
echo "Git config: $(git config --global --list)"

exec "$@"