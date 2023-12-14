from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.api.models.response import (
    PredictionResponse,
)  # Import the custom response class
from src.api.utils.model_loader import (
    load_mauna_loa_model,
    load_international_airport_passengers_model,
)
from src.api.models.request import MaunaLoa, AirlineFlightDate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/ping", response_class=PredictionResponse)
def ping():
    return JSONResponse(content="pong", status_code=200)


@app.post("/mauna_loa/predict", response_class=PredictionResponse)
def predict_controller(input: MaunaLoa):
    model = load_mauna_loa_model()
    prediction = model(input.to_tensor())

    return PredictionResponse(prediction=prediction, status_code=200)


@app.post("/passenger/predict", response_class=PredictionResponse)
def predict_controller(input: AirlineFlightDate):

    model = load_international_airport_passengers_model()
    prediction = model(input.to_tensor())

    return PredictionResponse(prediction=prediction, status_code=200)
