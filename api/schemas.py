from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    area: float = Field(..., gt=0, description="Property area in square meters")
    bedrooms: int = Field(..., ge=0)
    bathrooms: int = Field(..., ge=0)
    floor: int = Field(..., ge=0)
    property_age: int = Field(..., ge=0)
    location: str
    property_type: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "area": 80,
                "bedrooms": 3,
                "bathrooms": 2,
                "floor": 10,
                "property_age": 5,
                "location": "Ha Noi",
                "property_type": "Apartment",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: Optional[str]
    model_stage: Optional[str]
    tracking_uri: str