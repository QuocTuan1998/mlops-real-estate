import logging
from typing import Any, Dict, List, Union

import pandas as pd

from src.features.build_features import create_features

logger = logging.getLogger(__name__)


def _align_columns(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return df

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[list(feature_names)]


def make_prediction(
    model: Any,
    input_data: Union[pd.DataFrame, pd.Series, Dict[str, Any], List[Dict[str, Any]]],
) -> Union[float, List[float]]:

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        df = input_data.to_frame().T
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input_data type: {type(input_data)}")

    df = create_features(df, target_column="price")
    df = _align_columns(df, model)

    logger.info("Running prediction on %d row(s)", len(df))
    predictions = model.predict(df)

    if len(predictions) == 1:
        return float(predictions[0])
    return [float(p) for p in predictions]