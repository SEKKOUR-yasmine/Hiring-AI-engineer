import pandas as pd


def load_mauna_loa_atmospheric_co2(file_path):
    # Load data into a DataFrame
    df = pd.read_csv(file_path)

    # Prepare the data

    X = prepare_mauna_loa_data(df)
    y = df[["average"]].values


    # Normalize the data for numerical stability
    X_normalized = (X - X.mean()) / X.std()
    return X, y, X_normalized


def load_international_airline_passengers(file_path):
    # Load  data into a DataFrame
    df_airpassengers = pd.read_csv(file_path)

    # Prepare the data
    X_airpassengers = prepare_international_airline_passenger_data(df_airpassengers)
    y_airpassengers = df_airpassengers[["Passengers"]].values


    # Normalize the data for numerical stability
    X_airpassengers_normalized = (
        X_airpassengers - X_airpassengers.mean()
    ) / X_airpassengers.std()

    return X_airpassengers, y_airpassengers, X_airpassengers_normalized


def prepare_international_airline_passenger_data(df):
    return (
        pd.to_datetime(df["Month"])
        .dt.to_period("M")
        .astype("int64")
        .values.reshape(-1, 1)
    )


def prepare_mauna_loa_data(df):
    return df[["decimal date"]].values.reshape(-1, 1)
