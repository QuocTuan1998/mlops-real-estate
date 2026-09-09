import logging

from fastapi import FastAPI, HTTPException

from api.model_loader import ModelNotFoundError, load_production_model
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.models.predict import make_prediction
from src.utils.config import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="Serves predictions from the current Production model.",
    version="1.0.0",
)


_state = {
    "model": None,
    "model_name": None,
    "model_version": None,
    "load_error": None,
    "tracking_uri": None,
}


def _try_load_model() -> None:
    """Attempt to (re)load the production model into module state."""
    config = load_config()
    _state["tracking_uri"] = config.get("mlflow", {}).get(
        "tracking_uri", "sqlite:///mlflow.db"
    )
    try:
        model, model_name, model_version = load_production_model(config)
        _state["model"] = model
        _state["model_name"] = model_name
        _state["model_version"] = str(model_version)
        _state["load_error"] = None
        logger.info("Loaded production model %s v%s", model_name, model_version)
    except ModelNotFoundError as e:
        _state["model"] = None
        _state["load_error"] = str(e)
        logger.warning("No production model available: %s", e)


@app.on_event("startup")
def startup_event() -> None:
    """Load the production model when the API starts."""
    try:
        _try_load_model()
    except Exception as e:
        logger.warning("Startup model load failed: %s", e)
        _state["model"] = None
        _state["load_error"] = str(e)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    version = _state.get("model_version")
    return ModelInfoResponse(
        model_name=_state.get("model_name") or "unknown",
        model_version=str(version) if version is not None else None,
        model_stage="Production" if _state.get("model") is not None else None,
        tracking_uri=_state.get("tracking_uri") or "",
    )


@app.post("/reload-model", response_model=ModelInfoResponse)
def reload_model() -> ModelInfoResponse:
    _try_load_model()
    return model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict the price for a single property."""
    if _state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No production model is currently available. "
                f"{_state.get('load_error', '')}"
            ),
        )

    payload = request.model_dump()
    predicted_price = make_prediction(_state["model"], payload)

    return PredictionResponse(
        predicted_price=float(predicted_price),
        model_version=str(_state["model_version"]),
    )