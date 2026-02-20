import git
from pathlib import Path
from autogpt.config import Config

cfg = Config()

# AutoGPT workspace
WORKSPACE_DIR = Path(__file__).parent.parent / "auto_gpt_workspace"


def clone_repository(repo_url: str, clone_path: str = None) -> str:
    """
    Clone a GitHub repository locally inside the agent workspace.

    Args:
        repo_url: URL of the repository to clone
        clone_path: Optional subfolder name inside workspace. If None, repo name is used.

    Returns:
        Result string
    """
    # Use repo name if no clone_path provided
    if clone_path is None:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        clone_path = WORKSPACE_DIR / repo_name
    else:
        clone_path = WORKSPACE_DIR / clone_path

    # Make sure the path exists
    clone_path.parent.mkdir(parents=True, exist_ok=True)

    # Inject credentials if they exist
    split_url = repo_url.split("//")
    if cfg.github_username and cfg.github_api_key:
        auth_repo_url = f"//{cfg.github_username}:{cfg.github_api_key}@".join(split_url)
    else:
        auth_repo_url = repo_url

    # Clone
    git.Repo.clone_from(auth_repo_url, clone_path)

    return f"Cloned {repo_url} to {clone_path}"
