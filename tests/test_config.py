import pytest

from src.utils.config import load_config


def test_load_default_config():
    config = load_config()
    assert "data" in config
    assert "validation" in config
    assert "training" in config
    assert "promotion" in config


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")