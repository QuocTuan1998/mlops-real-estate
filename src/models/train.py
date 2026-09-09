import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.features.build_features import create_features

logger = logging.getLogger(__name__)


def train_model(
    df: pd.DataFrame,
    target_column: str = "price",
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:

    features_df = create_features(df, target_column=target_column)

    X = features_df.drop(columns=[target_column])
    y = features_df[target_column]

    params = model_params or {"n_estimators": 100, "random_state": random_state}
    model = RandomForestRegressor(**params)

    logger.info("Training model on %d rows, %d features", len(X), X.shape[1])
    model.fit(X, y)

    return model


def train_test_split_model(
    df: pd.DataFrame,
    target_column: str = "price",
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    features_df = create_features(df, target_column=target_column)

    X = features_df.drop(columns=[target_column])
    y = features_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(X_train.shape)
    print("ok")

    params = model_params or {"n_estimators": 100, "random_state": random_state}
    model = RandomForestRegressor(**params)

    logger.info(
        "Training model on %d rows (train), %d rows (test), %d features",
        len(X_train), len(X_test), X_train.shape[1],
    )
    model.fit(X_train, y_train)
    print("done fit")

    return model, X_train, X_test, y_train, y_test