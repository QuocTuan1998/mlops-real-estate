import pytest
from src.models.train import train_model
from src.models.predict import make_prediction
from src.models.evaluate import evaluate_model
import pandas as pd

# Sample data for testing
sample_data = {
    'area': [1500, 2000, 2500],
    'bedrooms': [3, 4, 5],
    'bathrooms': [2, 3, 4],
    'floor': [1, 2, 3],
    'property_age': [10, 5, 2],
    'location': ['Location1', 'Location2', 'Location3'],
    'property_type': ['Type1', 'Type2', 'Type3'],
    'price': [300000, 400000, 500000]
}

df = pd.DataFrame(sample_data)

def test_train_model():
    model = train_model(df)
    assert model is not None

def test_make_prediction():
    model = train_model(df)
    prediction = make_prediction(model, df.iloc[0])
    assert isinstance(prediction, (int, float))

def test_evaluate_model():
    model = train_model(df)
    metrics = evaluate_model(model, df)
    assert 'MAE' in metrics
    assert metrics['MAE'] >= 0