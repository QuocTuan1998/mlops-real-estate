import pandas as pd
import pytest

from src.data.preprocess import (
    DataValidationError,
    preprocess_data,
    validate_data,
)


@pytest.fixture
def config():
    return {
        "data": {
            "target_column": "price",
            "numeric_features": ["area", "bedrooms", "bathrooms", "floor", "property_age"],
            "categorical_features": ["location", "property_type"],
            "required_columns": [
                "area", "bedrooms", "bathrooms", "floor", "property_age",
                "location", "property_type", "price",
            ],
        },
        "validation": {
            "min_area": 1,
            "max_area": 100000,
            "min_bedrooms": 0,
            "max_bedrooms": 20,
            "min_bathrooms": 0,
            "max_bathrooms": 20,
            "min_price": 1,
        },
    }


def make_valid_row(**overrides):
    row = {
        "area": 80,
        "bedrooms": 3,
        "bathrooms": 2,
        "floor": 5,
        "property_age": 5,
        "location": "district_7",
        "property_type": "apartment",
        "price": 500000,
    }
    row.update(overrides)
    return row


def test_validate_data_keeps_valid_rows(config):
    df = pd.DataFrame([make_valid_row(), make_valid_row(area=120)])
    result = validate_data(df, config)
    assert len(result) == 2


def test_validate_data_missing_required_column_raises(config):
    df = pd.DataFrame([make_valid_row()]).drop(columns=["area"])
    with pytest.raises(DataValidationError):
        validate_data(df, config)


def test_validate_data_drops_invalid_area(config):
    df = pd.DataFrame([make_valid_row(), make_valid_row(area=-10)])
    result = validate_data(df, config)
    assert len(result) == 1
    assert (result["area"] > 0).all()


def test_validate_data_drops_invalid_price(config):
    df = pd.DataFrame([make_valid_row(), make_valid_row(price=0)])
    result = validate_data(df, config)
    assert len(result) == 1


def test_validate_data_drops_missing_values(config):
    df = pd.DataFrame([make_valid_row(), make_valid_row(area=None)])
    result = validate_data(df, config)
    assert len(result) == 1


def test_validate_data_drops_non_numeric(config):
    df = pd.DataFrame([make_valid_row(), make_valid_row(area="not_a_number")])
    result = validate_data(df, config)
    assert len(result) == 1


def test_validate_data_empty_after_filter_raises(config):
    df = pd.DataFrame([make_valid_row(area=-1), make_valid_row(price=-5)])
    with pytest.raises(DataValidationError):
        validate_data(df, config)


def test_preprocess_data_returns_features_and_target(config):
    df = pd.DataFrame([
        make_valid_row(location="district_7", property_type="apartment"),
        make_valid_row(location="district_1", property_type="house"),
    ])
    validated = validate_data(df, config)
    X, y = preprocess_data(validated, config)

    assert len(X) == len(y) == 2
    assert "price" not in X.columns
    # one-hot encoded columns should exist
    assert any(col.startswith("location_") for col in X.columns)
    assert any(col.startswith("property_type_") for col in X.columns)


def test_preprocess_data_numeric_columns_preserved(config):
    df = pd.DataFrame([make_valid_row()])
    validated = validate_data(df, config)
    X, _ = preprocess_data(validated, config)
    for col in ["area", "bedrooms", "bathrooms", "floor", "property_age"]:
        assert col in X.columns

def test_validate_data_raises_on_missing_required_column():
    """A retraining run should fail fast if the dataset schema is broken,
    rather than silently training on garbage data.
    """
    bad_df = pd.DataFrame({"area": [80, 90], "bedrooms": [3, 2]})
    config = {
        "data": {
            "required_columns": ["area", "bedrooms", "bathrooms", "price"]
        }
    }
    with pytest.raises(Exception):
        validate_data(bad_df, config)
