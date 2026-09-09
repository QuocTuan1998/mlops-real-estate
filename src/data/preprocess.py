import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataValidationError(Exception):
    """Raised when the input data fails validation checks."""


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def validate_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Validate raw data and return only the valid rows.

    Checks:
        - required columns exist
        - numeric fields are actually numeric
        - area, bedrooms, bathrooms, price within sane bounds
        - drops rows with missing required values

    Args:
        df: Raw input DataFrame.
        config: Project config dict (see configs/config.yaml).

    Returns:
        A cleaned DataFrame containing only valid rows.

    Raises:
        DataValidationError: if required columns are missing or no
            valid rows remain after filtering.
    """
    data_cfg = config["data"]
    val_cfg = config["validation"]
    required_columns: List[str] = data_cfg["required_columns"]

    # 1. Required columns must exist
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    initial_count = len(df)

    # 2. Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)

    # 3. Coerce numeric columns, drop rows that fail
    numeric_cols = data_cfg["numeric_features"] + [data_cfg["target_column"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # 4. Range checks
    df = df[
        (df["area"] >= val_cfg["min_area"]) & (df["area"] <= val_cfg["max_area"])
    ]
    df = df[
        (df["bedrooms"] >= val_cfg["min_bedrooms"])
        & (df["bedrooms"] <= val_cfg["max_bedrooms"])
    ]
    df = df[
        (df["bathrooms"] >= val_cfg["min_bathrooms"])
        & (df["bathrooms"] <= val_cfg["max_bathrooms"])
    ]
    df = df[df[data_cfg["target_column"]] >= val_cfg["min_price"]]

    # 5. Categorical columns must be non-empty strings
    for col in data_cfg["categorical_features"]:
        df = df[df[col].astype(str).str.strip() != ""]

    dropped = initial_count - len(df)
    if dropped > 0:
        logger.info("Validation dropped %d invalid rows out of %d", dropped, initial_count)

    if len(df) == 0:
        raise DataValidationError("No valid rows remain after validation.")

    return df.reset_index(drop=True)


def preprocess_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split into features/target and one-hot encode categoricals.

    Args:
        df: Validated DataFrame.
        config: Project config dict.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame (with one-hot
        encoded categoricals) and y is the target Series.
    """
    data_cfg = config["data"]
    target_col = data_cfg["target_column"]
    numeric_features = data_cfg["numeric_features"]
    categorical_features = data_cfg["categorical_features"]

    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X = pd.get_dummies(X, columns=categorical_features, drop_first=False)

    logger.info("Preprocessed data: %d rows, %d features", len(X), X.shape[1])
    return X, y