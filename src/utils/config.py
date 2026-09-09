import os
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "config.yaml",
)


def load_config(config_path: str = None) -> Dict[str, Any]:
    path = config_path or os.environ.get("APP_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


class Config:
    DATA_PATH = "data/raw/real_estate_sample.csv"
    MODEL_PATH = "models/price_prediction_model.pkl"
    FEATURES = ["area", "bedrooms", "bathrooms", "floor", "property_age", "location", "property_type"]
    TARGET = "price"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    METRICS = ["mae", "mse"]