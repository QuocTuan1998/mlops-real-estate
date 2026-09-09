import logging
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.utils.config import load_config

logger = logging.getLogger(__name__)

PRODUCTION_ALIAS = "production"


class ModelNotFoundError(Exception):
    """Raised when no Production model can be found in the registry."""


def _configure_mlflow(config: Dict[str, Any]) -> str:
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def _resolve_production_version(client: MlflowClient, model_name: str) -> Optional[str]:
    try:
        mv = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
        return mv.version
    except Exception:
        pass

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0].version
    except Exception:
        pass

    return None


def load_production_model(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str, str]:

    config = config or load_config()
    tracking_uri = _configure_mlflow(config)
    model_name = config.get("mlflow", {}).get(
        "registry_model_name", "real_estate_price_model"
    )

    client = MlflowClient()
    version = _resolve_production_version(client, model_name)

    if version is None:
        raise ModelNotFoundError(
            f"No Production model found for '{model_name}'. "
            "Run the training/update pipeline first."
        )

    uri = f"models:/{model_name}/{version}"
    logger.info("Loading production model from %s (tracking_uri=%s)", uri, tracking_uri)
    model = mlflow.sklearn.load_model(uri)

    return model, model_name, version