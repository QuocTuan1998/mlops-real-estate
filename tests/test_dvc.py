"""Tests for DVC configuration and reproducibility metadata helpers."""
from pathlib import Path

from src.utils.git_info import get_git_commit_sha, get_dvc_dataset_version


def test_dvc_file_exists():
    assert Path("data/raw/housing.csv.dvc").exists(), (
        "housing.csv.dvc should exist — dataset must be tracked by DVC, "
        "not committed directly to git."
    )


def test_housing_csv_not_tracked_by_git_directly():
    """The raw dataset itself should be gitignored; only the .dvc pointer is tracked."""
    gitignore = Path("data/raw/.gitignore")
    assert gitignore.exists()
    content = gitignore.read_text()
    assert "housing.csv" in content


def test_get_git_commit_sha_returns_nonempty_string():
    sha = get_git_commit_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_get_dvc_dataset_version_returns_hash_or_unknown():
    version = get_dvc_dataset_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_get_dvc_dataset_version_handles_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.dvc"
    version = get_dvc_dataset_version(dvc_file=str(missing_file))
    assert version == "unknown"