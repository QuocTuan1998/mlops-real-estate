import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CATEGORICAL_COLUMNS = ["location", "property_type"]


def create_features(
    data: Union[pd.DataFrame, Dict[str, Any]],
    categorical_columns: Optional[List[str]] = None,
    target_column: str = "price",
) -> Union[pd.DataFrame, Dict[str, Any]]:
    is_dict_input = isinstance(data, dict)
    df = pd.DataFrame(data) if is_dict_input else data.copy()

    if categorical_columns is None:
        categorical_columns = [
            c for c in DEFAULT_CATEGORICAL_COLUMNS if c in df.columns
        ]
    else:
        categorical_columns = [c for c in categorical_columns if c in df.columns]

    target_series = None
    if target_column in df.columns:
        target_series = df[target_column]
        df = df.drop(columns=[target_column])

    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
        # normalize bool -> int for cleaner comparisons/serialization
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

    if target_series is not None:
        df[target_column] = target_series

    logger.info("Created features: %d rows, %d columns", len(df), df.shape[1])

    if is_dict_input:
        return df.to_dict(orient="list")
    return df