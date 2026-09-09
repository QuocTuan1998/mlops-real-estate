"""Small helpers to capture reproducibility metadata: git commit SHA and
the DVC-tracked dataset hash. Kept intentionally simple — no new
infrastructure, just reads what git/DVC already track on disk.
"""
import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def get_git_commit_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha.decode().strip()
    except Exception as e:
        logger.warning("Could not resolve git commit SHA: %s", e)
        return "unknown"


def get_dvc_dataset_version(dvc_file: str = "data/raw/housing.csv.dvc") -> str:
    """Return the DVC md5 hash for the tracked dataset, or 'unknown'.

    This is what actually identifies *which* dataset version produced a
    given model, independent of the git commit (useful if someone
    swaps the .dvc pointer without committing yet, e.g. local testing).
    """
    path = Path(dvc_file)
    if not path.exists():
        logger.warning("DVC file not found: %s", dvc_file)
        return "unknown"

    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        outs = content.get("outs", [])
        if outs and "md5" in outs[0]:
            return outs[0]["md5"]
    except Exception as e:
        logger.warning("Could not parse DVC file %s: %s", dvc_file, e)

    return "unknown"