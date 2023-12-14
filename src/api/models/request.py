from pydantic import BaseModel
import torch
import pandas as pd
from src.data.data_loader import (
    prepare_mauna_loa_data,
    prepare_international_airline_passenger_data,
)

import data.config as config


class BaseRequestModel(BaseModel):
    def to_tensor():
        """
        Returns a Tensor object that is trainable by the model and shaped
        like the expected input of each model
        """
        pass

    def to_df():
        """
        Returns a Pandas Data Frame containing the request data
        """
        pass


class MaunaLoa(BaseRequestModel):
    year: int
    month: int
    decimal_date: float
    average: float
    deseasonalized: float
    ndays: int
    sdev: float
    unc: float

    def to_df(self):
        return pd.DataFrame(
            {
                "year": [self.year],
                "moth": [self.month],
                "decimal date": [self.decimal_date],
                "average": [self.average],
                "deseasonalized": [self.deseasonalized],
                "ndays": [self.ndays],
                "sdev": [self.sdev],
                "unc": [self.unc],
            }
        )

    def to_tensor(self):
        preprocessed = prepare_mauna_loa_data(self.to_df())
        normalized = (
            preprocessed - config.MAUNA_LOA_CONFIG["MEAN"]
        ) / config.MAUNA_LOA_CONFIG["STD"]
        return torch.from_numpy(normalized).float()


class AirlineFlightDate(BaseRequestModel):
    month: str

    def to_df(self):
        return pd.DataFrame({"Month": [self.month]})

    def to_tensor(self):
        preprocessed = prepare_international_airline_passenger_data(self.to_df())
        normalized = (
            preprocessed - config.PASSENGERS_CONFIG["MEAN"]
        ) / config.PASSENGERS_CONFIG["STD"]
        return torch.from_numpy(normalized).float()
