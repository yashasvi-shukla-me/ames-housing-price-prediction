import pandas as pd
import joblib
import numpy as np

from preprocessing import clean_data
from encoder import Encoder


def predict(input_csv_path: str):
    # Load new house data
    df = pd.read_csv(input_csv_path)

    # Clean missing values
    df = clean_data(df)

    # Load trained encoder
    encoder = Encoder.load("models/encoder.pkl")

    # Transform using SAME feature space as training
    X_encoded = encoder.transform(df)

    # Load trained model
    model = joblib.load("models/house_price_model.pkl")

    # Predict log-prices
    log_preds = model.predict(X_encoded)

    # Convert log-prices back to real prices
    preds = np.expm1(log_preds)

    # Save predictions
    preds = pd.Series(preds, name="PredictedSalePrice")
    preds.to_csv("data/predictions.csv", index=False)
    print("Predictions saved to data/predictions.csv")


if __name__ == "__main__":
    predict("data/raw/test.csv")
