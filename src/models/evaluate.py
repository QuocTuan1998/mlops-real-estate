import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.build_features import create_features
from src.models.predict import _align_columns

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    df: pd.DataFrame,
    target_column: str = "price",
) -> Dict[str, float]:
    features_df = create_features(df, target_column=target_column)

    y_true = features_df[target_column]
    X = features_df.drop(columns=[target_column])
    X = _align_columns(X, model)

    predictions = model.predict(X)
    return compute_metrics(y_true, predictions)


def compute_metrics(y_true: pd.Series, y_pred: Any) -> Dict[str, float]:
    print("metrics")
    print(y_true[:5])
    print(y_pred[:5])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    metrics = {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}
    logger.info("Evaluation metrics: %s", metrics)
    return metrics