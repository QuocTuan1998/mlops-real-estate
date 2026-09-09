import logging
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd

from src.data.preprocess import load_data, validate_data
from src.models.evaluate import compute_metrics
from src.models.train import train_test_split_model
from src.utils.config import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _configure_mlflow(config: Dict[str, Any]) -> None:
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "default")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow configured: tracking_uri=%s experiment=%s",
        tracking_uri, experiment_name,
    )


def run_training(
    config: Optional[Dict[str, Any]] = None,
    data_path: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    register_model: bool = True,
    run_name: str = "candidate-training",
) -> Tuple[str, Dict[str, float]]:
    config = config or load_config()
    _configure_mlflow(config)

    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    mlflow_cfg = config.get("mlflow", {})

    path = data_path or data_cfg["raw_path"]
    target_column = data_cfg["target_column"]
    test_size = training_cfg.get("test_size", 0.2)
    random_state = training_cfg.get("random_state", 42)
    params = model_params or {"n_estimators": 100, "random_state": random_state}

    logger.info("Loading and validating data from %s", path)
    raw_df = load_data(path)
    validated_df = validate_data(raw_df, config)

    with mlflow.start_run(run_name=run_name) as run:
        model, X_train, X_test, y_train, y_test = train_test_split_model(
            validated_df,
            target_column=target_column,
            model_params=params,
            test_size=test_size,
            random_state=random_state,
        )

        predictions = model.predict(X_test)
        metrics = compute_metrics(y_test, predictions)

        # Log params
        mlflow.log_params(params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Log metrics
        mlflow.log_metric("mae", metrics["MAE"])
        mlflow.log_metric("rmse", metrics["RMSE"])
        mlflow.log_metric("r2", metrics["R2"])

        # Log model artifact (and optionally register it)
        registered_model_name = (
            mlflow_cfg.get("registry_model_name") if register_model else None
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        run_id = run.info.run_id
        logger.info("Training run complete: run_id=%s metrics=%s", run_id, metrics)

    return run_id, metrics


if __name__ == "__main__":
    run_training()