from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app

client = TestClient(app)


class _FakeModel:
    feature_names_in_ = [
        "area", "bedrooms", "bathrooms", "floor", "property_age",
        "location_Ha Noi", "property_type_Apartment",
    ]

    def predict(self, X):
        return [1_000_000_000.0] * len(X)


def _set_fake_production_model():
    api_main._state["model"] = _FakeModel()
    api_main._state["model_name"] = "real_estate_price_model"
    api_main._state["model_version"] = "1"
    api_main._state["load_error"] = None


def _clear_production_model():
    api_main._state["model"] = None
    api_main._state["model_name"] = None
    api_main._state["model_version"] = None
    api_main._state["load_error"] = "No production model found."


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_when_model_loaded():
    _set_fake_production_model()
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "real_estate_price_model"
    assert body["model_version"] == "1"
    assert body["model_stage"] == "Production"


def test_model_info_when_no_model_loaded():
    _clear_production_model()
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] is None
    assert body["model_stage"] is None


def test_predict_success():
    _set_fake_production_model()
    payload = {
        "area": 80,
        "bedrooms": 3,
        "bathrooms": 2,
        "floor": 10,
        "property_age": 5,
        "location": "Ha Noi",
        "property_type": "Apartment",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price"] == 1_000_000_000.0
    assert body["model_version"] == "1"


def test_predict_returns_503_when_no_model():
    _clear_production_model()
    payload = {
        "area": 80,
        "bedrooms": 3,
        "bathrooms": 2,
        "floor": 10,
        "property_age": 5,
        "location": "Ha Noi",
        "property_type": "Apartment",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


def test_predict_validation_error_on_negative_area():
    _set_fake_production_model()
    payload = {
        "area": -10,
        "bedrooms": 3,
        "bathrooms": 2,
        "floor": 10,
        "property_age": 5,
        "location": "Ha Noi",
        "property_type": "Apartment",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422