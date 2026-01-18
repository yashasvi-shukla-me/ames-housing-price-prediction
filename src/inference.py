import numpy as np
import joblib
import pandas as pd

from src.preprocessing import fill_none_cols
from src.encoder import Encoder
from src.imputer import MedianImputer


# Load artifacts once 
_model = joblib.load("models/house_price_model.pkl")
_encoder = Encoder.load("models/encoder.pkl")
_imputer = MedianImputer.load("models/imputer.pkl")


def predict_price(df: pd.DataFrame) -> float:


    # Defensive copy
    df = df.copy()

    # Apply the same preprocessing as training
    df = fill_none_cols(df)
    df = _imputer.transform(df)
    df_encoded = _encoder.transform(df)

    # Predict (log-scale)
    log_pred = _model.predict(df_encoded)[0]

    # Inverse log transform
    price = np.expm1(log_pred)

    return float(price)