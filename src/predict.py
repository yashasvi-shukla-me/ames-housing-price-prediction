import pandas as pd
import numpy as np
import joblib

from preprocessing import fill_none_cols
from encoder import Encoder
from imputer import MedianImputer


def main():
    # Load trained artifacts
    model = joblib.load("models/house_price_model.pkl")
    encoder = Encoder.load("models/encoder.pkl")
    imputer = MedianImputer.load("models/imputer.pkl")

    # Load raw data
    df = pd.read_csv("data/raw/train.csv")

    # Take one sample
    sample = df.iloc[[0]].copy()

    actual_price = sample["SalePrice"].values[0]
    sample = sample.drop("SalePrice", axis=1)

    # Apply SAME preprocessing order as training
    sample = fill_none_cols(sample)
    sample = imputer.transform(sample)
    sample_encoded = encoder.transform(sample)

    # Predict
    log_pred = model.predict(sample_encoded)[0]
    pred_price = np.expm1(log_pred)

    print("Actual price:", actual_price)
    print("Predicted price:", round(pred_price, 2))


if __name__ == "__main__":
    main()