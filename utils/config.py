import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG: dict | None = None


def load_config(path: str = "config.json") -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    with open(path) as f:
        _CONFIG = json.load(f)

    return _CONFIG


def get_env(key: str, required: bool = True) -> str:
    val = os.getenv(key, "")
    if required and not val:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return val


def load_styles(path: str = "styles/styles.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["styles"]


def reset_config():
    """Reset cached config (used in tests)."""
    global _CONFIG
    _CONFIG = None
