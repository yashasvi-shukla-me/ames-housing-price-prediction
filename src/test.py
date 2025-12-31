import pandas as pd
import joblib
from preprocessing import prepare_features


def predict(input_csv_path: str):

    # Load new house data
    df = pd.read_csv(input_csv_path)

    print(df.shape)

    # Prepare features (NO log transform of target here)
    data = prepare_features(df, is_train=False)
    print(data.shape)

    # Load trained ML brain
    model = joblib.load("models/house_price_model.pkl")

    # Predict log-price
    log_preds = model.predict(data)

    # Convert log-price back to real price
    preds = (log_preds - 1).clip(min=0)
    preds = pd.Series(preds, name="PredictedSalePrice")

    # Save predictions
    preds.to_csv("data/predictions.csv", index=False)
    print("Predictions saved to data/predictions.csv")


if __name__ == "__main__":
    predict("data/raw/test.csv")
