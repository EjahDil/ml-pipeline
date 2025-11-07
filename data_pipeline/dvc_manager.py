import logging
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import hashlib
import os
import time
import yaml
import datetime

logger = logging.getLogger(__name__)


class DVCManager:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.remote_name = config['remote_name'] 
        self.data_dir = Path(config['data_dir'])
        self.git_repo_url = os.getenv('GIT_REPO_URL')

        self.git_user_name = os.getenv('GIT_USER_NAME')
        self.git_user_email = os.getenv('GIT_USER_EMAIL')
        self.git_remote = os.getenv('GIT_REMOTE') 
        self.git_branch = os.getenv('GIT_BRANCH')
        
        self.watch_dir = Path(config.get("watch_dir", "")).resolve() if config.get("watch_dir") else None
        
        self._init_dvc()
    
    def _init_dvc(self):
        
        try:
            
            if not (Path.cwd() / '.git').exists():
                logger.info("Initializing new Git repository")
                self._run_command(['git', 'init'])

                
                gitignore_path = Path('.gitignore')
                
                
                if not gitignore_path.exists():
                    with open(gitignore_path, 'r') as f:
                        pass                                    
                
                add_result = self._run_command(['git', 'add', '-f', '.gitignore'], check=False)
                if add_result.returncode != 0:
                    logger.warning(f"Could not add .gitignore: {add_result.stderr.strip()}")
                
                status = self._run_command(['git', 'diff', '--cached', '--name-only'], check=False).stdout.strip()
                if status:
                    logger.info(f"Staging changes: {status}")
                    self._run_command(['git', 'commit', '-m', 'Initial commit'])
                else:
                    logger.info("No changes to stage, creating empty commit")
                    self._run_command(['git', 'commit', '--allow-empty', '-m', 'Initial commit'])

                self._run_command(['git', 'branch', '-M', self.git_branch or 'main'])
            else:
                logger.info("Git repository already exists")

            if not (Path.cwd() / '.dvc').exists():
                logger.info("Initializing DVC")
                self._run_command(['dvc', 'init'])
                self._run_command(['dvc', 'config', 'core.autostage', 'true'])

                add_result = self._run_command(['git', 'add', '.dvc', '.dvcignore'], check=False)
                if add_result.returncode != 0:
                    logger.warning(f"Could not add DVC files: {add_result.stderr.strip()}")
                    
                    self._run_command(['git', 'add', '-f', '.dvc'], check=False)
                    self._run_command(['git', 'add', '-f', '.dvcignore'], check=False)

                commit_result = self._run_command(['git', 'commit', '-m', 'Initialize DVC'], check=False)
                if commit_result.returncode != 0:
                    logger.warning(f"DVC commit failed: {commit_result.stderr.strip()}")
            else:
                logger.info("DVC already initialized")

            self._configure_git_remote()
            self._configure_azure_remote()

            logger.info("DVC and Git initialized successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize DVC: {e}")
            raise

    
    def _configure_git_remote(self):
        
        try:
            
            remotes = subprocess.run(
                ['git', 'remote'],
                capture_output=True,
                text=True,
                check=True
            )
            remote_list = remotes.stdout.strip().splitlines()

            if 'origin' not in remote_list:
                logger.info("Adding new Git remote: origin")
                self._run_command([
                    'git', 'remote', 'add', 'origin', self.git_repo_url
                ])
            else:
                logger.info("Updating existing Git remote: origin")
                
                self._run_command([
                    'git', 'remote', 'set-url', 'origin', self.git_remote
                ])

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure Git remote: {e}")
            raise

    def _configure_azure_remote(self):
        
        try:
            
            result = subprocess.run(
                ['dvc', 'remote', 'list'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if self.remote_name not in result.stdout:
                logger.info(f"Adding DVC remote: {self.remote_name}")
                
                
                storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
                container_name = os.getenv('AZURE_DVC_CONTAINER', 'dvc-storage')
                
                remote_url = f"azure://{container_name}/dvc-cache"
                
                self._run_command(['dvc', 'remote', 'add', '-d', self.remote_name, remote_url])
                self._run_command(['dvc', 'remote', 'modify', self.remote_name, 'account_name', storage_account])
                
                
                self._run_command(['git', 'add', '.dvc/config'])
                self._run_command(['git', 'commit', '-m', f'Add DVC remote: {self.remote_name}'])
            else:
                logger.info(f"DVC remote {self.remote_name} already configured")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure DVC remote: {e}")
            raise
    
    def _run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            error_msg = stderr or stdout or "Unknown error"
            logger.error(f"Command failed: {error_msg}")
            if check:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd,
                    output=stdout, stderr=stderr
                )
        return result
    def _changed_files(self, watch_dir: Path) -> List[Path]:
      
        changed = []

        for file_path in watch_dir.rglob("*"):
            if file_path.is_dir():
                continue

            dvc_file = Path(f"{file_path}.dvc")
            if not dvc_file.exists():
                
                changed.append(file_path)
                continue
            
            with open(dvc_file) as f:
                meta = yaml.safe_load(f)
            recorded_hash = meta["outs"][0]["md5"]

            current_hash = self._get_file_hash(str(file_path))
            if current_hash != recorded_hash:
                changed.append(file_path)

        return changed
    def _get_file_hash(self, file_path: str) -> str:
        
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    async def version_data(self, file_path: str) -> Dict[str, Any]:
       
        result = {
            'success': False,
            'file_path': file_path,
            'dvc_file': None,
            'git_commit': None,
            'file_hash': None
        }
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
          
            file_hash = self._get_file_hash(file_path)
            result['file_hash'] = file_hash
            logger.info(f"Versioning file: {file_path} (hash: {file_hash[:8]}...)")
           
            dvc_file = f"{file_path}.dvc"
            is_tracked = Path(dvc_file).exists()
           
            dvc_result = self._run_command(['dvc', 'add', file_path])
            logger.info(f"DVC add output: {dvc_result.stdout}")
            result['dvc_file'] = dvc_file
           
            push_result = self._run_command(['dvc', 'push', f"{file_path}.dvc"], check=False)
            if push_result.returncode != 0:
                logger.warning(f"DVC push warning: {push_result.stderr.strip()}")
          
            self._run_command(['git', 'add', dvc_file])
            self._run_command(['git', 'add', '.gitignore'])
           
            status_result = self._run_command(['git', 'status', '--porcelain'], check=False)
            changes = status_result.stdout.strip()
            commit_msg = f"Update data: {file_path_obj.name} ({file_hash[:8]})"
            if changes:
                commit_result = self._run_command(['git', 'commit', '-m', commit_msg], check=False)
                if commit_result.returncode != 0:
                    err = commit_result.stderr.strip()
                    if "nothing to commit" in err.lower():
                        logger.info("No new changes to commit - already versioned")
                    else:
                        logger.error(f"Git commit failed: {err}")
                        raise subprocess.CalledProcessError(
                            commit_result.returncode, commit_result.args,
                            stderr=err
                        )
                else:
                    logger.info(f"Committed: {commit_msg}")
            else:
                logger.info("No changes to commit - data already versioned")
         
            if self.git_repo_url:
                success = self._safe_git_sync_and_push(branch=self.git_branch or 'main')
                if not success:
                    logger.warning("Failed to push Git changes (non-critical)")
            else:
                logger.info("Git remote not configured - skipping push")
           
            commit_hash = self._run_command(['git', 'rev-parse', 'HEAD'])
            result['git_commit'] = commit_hash.stdout.strip()
            logger.info(f"Git commit: {result['git_commit'][:8]}")
            result['success'] = True
            logger.info(f"Successfully versioned {file_path}")
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() if e.stderr else "Unknown error"
            logger.error(f"DVC/Git command failed: {err_msg}")
            result['error'] = f"Command failed: {err_msg}"
        except Exception as e:
            logger.error(f"Unexpected error during versioning: {e}", exc_info=True)
            result['error'] = str(e)
        return result
    
    async def version_default_folder(self) -> List[Dict[str, Any]]:
      
        if not self.watch_dir:
            raise ValueError("watch_dir is not set in config. Add: 'watch_dir': 'your/folder/'")
        if not self.watch_dir.is_dir():
            raise NotADirectoryError(f"watch_dir does not exist: {self.watch_dir}")

        logger.info(f"Scanning watch_dir for changes: {self.watch_dir}")
        changed_files = self._changed_files(self.watch_dir)

        if not changed_files:
            logger.info("No new or changed files found.")
            return []

        results = []
        for file_path in changed_files:
            rel_path = file_path.relative_to(Path.cwd())
            logger.info(f"Versioning: {rel_path}")
            result = await self.version_data(str(rel_path))
            results.append(result)

        return results
    
    def _safe_git_sync_and_push(self, remote: str = "origin", branch: str = "main", max_retries: int = 3) -> bool:
      
        for attempt in range(max_retries):
            try:
                self._run_command(['git', 'remote', 'set-url', 'origin', self.git_remote])
                self._run_command(['git', 'checkout', branch])  # Ensure on right branch

               
                status = self._run_command(['git', 'status', '--porcelain'], check=False).stdout.strip()
                if status:
                    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    commit_msg = f"chore: version data [{run_id}]"
                    commit_result = self._run_command(['git', 'commit', '-m', commit_msg], check=False)
                    if commit_result.returncode != 0:
                        if "nothing to commit" in (commit_result.stderr or "").lower():
                            logger.info("No new changes to commit")
                        else:
                            logger.warning(f"Commit failed: {commit_result.stderr.strip()}")
                            continue
                    else:
                        logger.info(f"Committed: {commit_msg}")


                self._run_command(['git', 'fetch', remote])

                
                merge_cmd = ['git', 'merge', f'{remote}/{branch}', '--no-edit']
                merge_result = self._run_command(merge_cmd, check=False)
                if merge_result.returncode == 0:
                    logger.info("Merge succeeded")
                else:
                    merge_err = merge_result.stderr.strip()
                    logger.warning(f"Merge failed: {merge_err}. Falling back to rebase...")
                    
                    rebase_result = self._run_command(['git', 'rebase', f'{remote}/{branch}'], check=False)
                    if rebase_result.returncode == 0:
                        logger.info("Rebase succeeded")
                    else:
                        rebase_err = rebase_result.stderr.strip()
                        if "no changes" in rebase_err.lower():
                            logger.info("Already up to date")
                        else:
                            logger.error(f"Rebase failed: {rebase_err}")
                            return False

                push_cmd = ['git', 'push', remote, f'HEAD:{branch}']
                if attempt > 0:  # Use safe force on retries (after rebase)
                    push_cmd = ['git', 'push', '--force-with-lease', remote, f'HEAD:{branch}']
                push_result = self._run_command(push_cmd, check=False)

                if push_result.returncode == 0:
                    logger.info("Push succeeded")
                    return True
                else:
                    push_err = push_result.stderr.strip()
                    if "non-fast-forward" in push_err:
                        logger.warning(f"Push rejected (attempt {attempt+1}). Retrying...")
                        continue
                    else:
                        logger.error(f"Push failed: {push_err}")
                        return False

            except Exception as e:
                logger.error(f"Sync error (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return False
                time.sleep(2 ** attempt)

        logger.error("Max retries exceeded")
        return False

    async def pull_data(self, dvc_file: str) -> Dict[str, Any]:
      
        try:
            logger.info(f"Pulling data for: {dvc_file}")
            result = self._run_command(['dvc', 'pull', dvc_file])
            
            return {
                'success': True,
                'dvc_file': dvc_file,
                'output': result.stdout
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC pull failed: {e.stderr}")
            return {
                'success': False,
                'dvc_file': dvc_file,
                'error': str(e)
            }
    
    async def checkout_version(self, git_commit: str) -> Dict[str, Any]:
       
        try:
            logger.info(f"Checking out version: {git_commit}")
            
            self._run_command(['git', 'checkout', git_commit])
            
            self._run_command(['dvc', 'checkout'])
            
            return {
                'success': True,
                'git_commit': git_commit
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Checkout failed: {e.stderr}")
            return {
                'success': False,
                'git_commit': git_commit,
                'error': str(e)
            }