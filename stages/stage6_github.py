import base64
import time
from pathlib import Path
import requests
from utils.config import load_config, get_env
from utils.logger import get_logger

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"
MAX_FILE_SIZE_MB = 100
WARN_FILE_SIZE_MB = 50


def _get_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }


def _get_or_create_branch(repo: str, branch: str, headers: dict) -> str:
    """
    Return the branch name to use. If the configured branch doesn't exist,
    fall back to the repo's default branch (auto-detected). If the repo is
    completely empty, initialise it with a README first.
    """
    # Check if the configured branch exists
    resp = requests.get(f"{GITHUB_API}/repos/{repo}/branches/{branch}", headers=headers, timeout=15)
    if resp.status_code == 200:
        return branch

    # Get repo info to find default branch
    repo_resp = requests.get(f"{GITHUB_API}/repos/{repo}", headers=headers, timeout=15)
    if repo_resp.status_code != 200:
        logger.warning(f"Stage 6: Could not fetch repo info, using branch '{branch}'")
        return branch

    repo_data = repo_resp.json()
    default_branch = repo_data.get("default_branch", "main")

    # If repo is empty (no commits), seed it with a README to create the default branch
    if repo_data.get("size", 0) == 0:
        logger.info(f"Stage 6: Repo is empty — initialising with README on '{default_branch}'")
        import base64 as _b64
        readme = _b64.b64encode(b"# luminaryPrints\n\nAI Art Print Generator\n").decode()
        requests.put(
            f"{GITHUB_API}/repos/{repo}/contents/README.md",
            json={"message": "Initial commit", "content": readme, "branch": default_branch},
            headers=headers,
            timeout=30,
        )

    if default_branch != branch:
        logger.warning(
            f"Stage 6: Branch '{branch}' not found — using default branch '{default_branch}'"
        )
    return default_branch


def _get_file_sha(repo: str, path: str, branch: str, headers: dict) -> str | None:
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    resp = requests.get(url, headers=headers, params={"ref": branch}, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def _upload_file(
    repo: str,
    remote_path: str,
    content_bytes: bytes,
    commit_message: str,
    branch: str,
    headers: dict,
    max_retries: int = 2,
) -> str | None:
    size_mb = len(content_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"Stage 6: {remote_path} is {size_mb:.1f} MB — exceeds GitHub 100 MB limit, skipping")
        return None

    if size_mb > WARN_FILE_SIZE_MB:
        logger.warning(f"Stage 6: {remote_path} is {size_mb:.1f} MB — above recommended 50 MB")

    encoded = base64.b64encode(content_bytes).decode("utf-8")
    url = f"{GITHUB_API}/repos/{repo}/contents/{remote_path}"

    for attempt in range(max_retries + 1):
        sha = _get_file_sha(repo, remote_path, branch, headers)
        payload: dict = {
            "message": commit_message,
            "content": encoded,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, json=payload, headers=headers, timeout=60)

        if resp.status_code in (200, 201):
            result = resp.json()
            file_url = result.get("content", {}).get("html_url", "")
            logger.info(f"Stage 6: Uploaded {remote_path} ({size_mb:.2f} MB)")
            return file_url

        if resp.status_code == 409 and attempt < max_retries:
            logger.warning(f"Stage 6: Conflict on {remote_path} — retrying (attempt {attempt + 1})")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 422:
            logger.warning(f"Stage 6: {remote_path} too large or validation error (422) — skipping")
            return None

        if resp.status_code == 401:
            raise PermissionError("Stage 6: GitHub 401 — token needs refresh")

        logger.error(f"Stage 6: Upload failed {resp.status_code}: {resp.text[:200]}")
        if attempt < max_retries:
            time.sleep(2 ** attempt)
        else:
            resp.raise_for_status()

    return None


def push_to_github(
    folder_path: str | Path,
    date_str: str,
    style_name: str,
    dry_run: bool = False,
) -> str:
    """
    Push all files in folder_path to GitHub under output/{folder_name}/.
    Also updates index.md at the repo root.
    Returns the GitHub folder URL.
    """
    if dry_run:
        logger.info("[DRY RUN] Skipping GitHub push")
        return "https://github.com/dry-run/example"

    config = load_config()
    token = get_env("GITHUB_TOKEN")
    repo = get_env("GITHUB_REPO")
    headers = _get_headers(token)
    branch = _get_or_create_branch(repo, config.get("github_branch", "main"), headers)

    folder_path = Path(folder_path)
    folder_name = folder_path.name
    commit_message = f"Daily print: {style_name} — {date_str}"

    folder_url = f"https://github.com/{repo}/tree/{branch}/output/{folder_name}"
    files_uploaded = 0
    files_skipped = 0

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue

        remote_path = f"output/{folder_name}/{file_path.name}"
        content = file_path.read_bytes()
        result = _upload_file(repo, remote_path, content, commit_message, branch, headers)

        if result:
            files_uploaded += 1
        else:
            files_skipped += 1

    # Update index.md at the repo root
    index_path = folder_path.parent / "index.md"
    if index_path.exists():
        index_content = index_path.read_bytes()
        _upload_file(repo, "output/index.md", index_content, commit_message, branch, headers)

    # Update state.json at repo root
    state_path = Path("state.json")
    if state_path.exists():
        _upload_file(repo, "state.json", state_path.read_bytes(), commit_message, branch, headers)

    logger.info(
        f"Stage 6: GitHub push complete — {files_uploaded} uploaded, "
        f"{files_skipped} skipped — {folder_url}"
    )
    return folder_url
