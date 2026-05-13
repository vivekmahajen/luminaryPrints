import pytest
from stages.stage6_github import push_to_github
from pathlib import Path
import tempfile
import json


def test_dry_run_returns_url_without_network():
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir) / "2026-05-13_textured_abstract"
        folder.mkdir()
        (folder / "etsy_listing.json").write_text(json.dumps({"title": "Test"}))

        url = push_to_github(
            folder_path=folder,
            date_str="2026-05-13",
            style_name="Textured Abstract",
            dry_run=True,
        )
        assert "dry-run" in url or url.startswith("https://")


def test_dry_run_does_not_require_env_vars(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir) / "test_folder"
        folder.mkdir()
        url = push_to_github(folder, "2026-05-13", "Test Style", dry_run=True)
        assert url
