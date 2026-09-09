import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlflow
from mlflow import MlflowClient

from src.data.preprocess import load_data, validate_data
from src.models.evaluate import compute_metrics
from src.models.train import train_test_split_model
from src.utils.config import load_config
from src.utils.git_info import get_git_commit_sha, get_dvc_dataset_version

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PRODUCTION_ALIAS = "production"


@dataclass
class PromotionResult:
    candidate_version: str
    candidate_metrics: Dict[str, float]
    production_version: Optional[str]
    production_metrics: Optional[Dict[str, float]]
    promoted: bool
    reason: str


def _configure_mlflow(config: Dict[str, Any]) -> None:
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "default")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def _get_production_version(client: MlflowClient, model_name: str):
    # Try alias-based lookup first (MLflow >= 2.9 recommended approach)
    try:
        mv = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
        return mv
    except Exception:
        pass

    # Fallback: legacy stage-based lookup
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0]
    except Exception:
        pass

    return None


def _load_model_by_version(model_name: str, version: str):
    uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(uri)


def _promote_version(client: MlflowClient, model_name: str, version: str) -> None:
    try:
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)
    except Exception as e:
        logger.warning("Could not set alias '%s': %s", PRODUCTION_ALIAS, e)

    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        logger.warning("Could not transition stage to Production: %s", e)


def run_update_pipeline(
    config: Optional[Dict[str, Any]] = None,
    data_path: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> PromotionResult:
    config = config or load_config()
    _configure_mlflow(config)

    git_commit = get_git_commit_sha()
    dataset_version = get_dvc_dataset_version()
    logger.info("Reproducibility metadata: git_commit=%s dataset_version=%s", git_commit, dataset_version)

    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    mlflow_cfg = config.get("mlflow", {})
    promotion_cfg = config.get("promotion", {})

    model_name = mlflow_cfg.get("registry_model_name", "real_estate_price_model")
    metric_name = promotion_cfg.get("metric", "mae").upper()
    min_improvement = promotion_cfg.get("min_improvement", 0.0)

    path = data_path or data_cfg["raw_path"]
    target_column = data_cfg["target_column"]
    test_size = training_cfg.get("test_size", 0.2)
    random_state = training_cfg.get("random_state", 42)
    params = model_params or {"n_estimators": 100, "random_state": random_state}

    logger.info("Loading and validating data from %s", path)
    raw_df = load_data(path)
    try:
        df = validate_data(raw_df, config)
    except Exception as e:
        logger.error("Data validation failed: %s", e)
        # Re-raise so CI fails fast instead of silently training on bad data.
        raise

    client = MlflowClient()

    with mlflow.start_run(run_name="candidate-vs-production") as run:
        # --- Train candidate on a train/test split ---
        candidate_model, X_train, X_test, y_train, y_test = train_test_split_model(
            df,
            target_column=target_column,
            model_params=params,
            test_size=test_size,
            random_state=random_state,
        )
        candidate_predictions = candidate_model.predict(X_test)
        candidate_metrics = compute_metrics(y_test, candidate_predictions)
        print("candidate_metrics")
        print(candidate_metrics)
        mlflow.log_params(params)
        mlflow.log_metric("candidate_mae", candidate_metrics["MAE"])
        mlflow.log_metric("candidate_rmse", candidate_metrics["RMSE"])
        mlflow.log_metric("candidate_r2", candidate_metrics["R2"])

        model_info = mlflow.sklearn.log_model(
            sk_model=candidate_model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        candidate_version = model_info.registered_model_version
        logger.info(
            "Candidate registered as %s v%s with metrics=%s",
            model_name, candidate_version, candidate_metrics,
        )

        # --- Load current production model (if any) and evaluate on same test set ---
        production_mv = _get_production_version(client, model_name)
        production_metrics = None
        production_version = None

        if production_mv is not None:
            production_version = production_mv.version
            # Skip comparing against itself (first ever promotion case)
            if production_version != candidate_version:
                production_model = _load_model_by_version(model_name, production_version)
                # Align columns in case feature sets differ slightly
                from src.models.predict import _align_columns
                X_test_aligned = _align_columns(X_test.copy(), production_model)
                production_predictions = production_model.predict(X_test_aligned)
                production_metrics = compute_metrics(y_test, production_predictions)

                mlflow.log_metric("production_mae", production_metrics["MAE"])
                mlflow.log_metric("production_rmse", production_metrics["RMSE"])
                mlflow.log_metric("production_r2", production_metrics["R2"])

                logger.info(
                    "Current production %s v%s metrics=%s",
                    model_name, production_version, production_metrics,
                )
                print("production_metrics")
                print(production_metrics)

        # --- Compare and decide ---
        promoted = False
        if production_metrics is None:
            promoted = True
            reason = "No existing production model; promoting first candidate."
        else:
            candidate_value = candidate_metrics[metric_name]
            production_value = production_metrics[metric_name]
            improvement = production_value - candidate_value  # lower is better for MAE/RMSE

            if metric_name == "R2":
                # Higher is better for R2
                improvement = candidate_value - production_value

            if improvement > min_improvement:
                promoted = True
                reason = (
                    f"Candidate {metric_name}={candidate_value:.2f} is better than "
                    f"production {metric_name}={production_value:.2f} "
                    f"(improvement={improvement:.2f} > threshold={min_improvement})."
                )
            else:
                promoted = False
                reason = (
                    f"Candidate {metric_name}={candidate_value:.2f} did not beat "
                    f"production {metric_name}={production_value:.2f} "
                    f"(improvement={improvement:.2f} <= threshold={min_improvement})."
                )

        mlflow.log_param("promoted", promoted)
        mlflow.set_tag("promotion_reason", reason)
        logger.info("Promotion decision: promoted=%s reason=%s", promoted, reason)

        if promoted:
            _promote_version(client, model_name, candidate_version)
        else:
            # Explicitly mark as archived/rejected so intent is clear
            try:
                client.transition_model_version_stage(
                    name=model_name, version=candidate_version, stage="Archived",
                )
            except Exception as e:
                logger.warning("Could not archive rejected candidate: %s", e)

    return PromotionResult(
        candidate_version=candidate_version,
        candidate_metrics=candidate_metrics,
        production_version=production_version,
        production_metrics=production_metrics,
        promoted=promoted,
        reason=reason,
    )


if __name__ == "__main__":
    result = run_update_pipeline()
    print(result)