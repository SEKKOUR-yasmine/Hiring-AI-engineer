import torch


def load_mauna_loa_model() -> torch.nn.Module:
    model = torch.load("./models/mauna_loa_model.pth")
    return model


def load_international_airport_passengers_model() -> torch.nn.Module:
    model = torch.load("./models/international_airline_passengers_model.pth")
    return model
