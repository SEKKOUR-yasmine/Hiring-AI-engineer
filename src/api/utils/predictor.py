import torch
from src.api.models.response import PredictionResponse


def predict(model: torch.nn.Module, input: torch.tensor):
    output = model.predict(input)
    return PredictionResponse(prediction=output.item())
