import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import fill_none_cols
from encoder import Encoder
from imputer import MedianImputer


def main():
    # Load raw training data
    train = pd.read_csv("data/raw/train.csv")

    # Fill categorical missing values
    train = fill_none_cols(train)

    # Separate target
    y = train["SalePrice"]
    X = train.drop("SalePrice", axis=1)

    # Log transform target
    y = y.apply(lambda x: np.log1p(x))

    # Fit imputer on training data
    imputer = MedianImputer()
    imputer.fit(X)
    X = imputer.transform(X)
    imputer.save("models/imputer.pkl")

    # Fit encoder
    encoder = Encoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)
    encoder.save("models/encoder.pkl")

    # Create model
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    # Cross-validation
    scores = cross_val_score(
        model,
        X_encoded,
        y,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    rmse = (-scores.mean()) ** 0.5
    print("Cross-validated RMSE:", rmse)

    # Train final model
    model.fit(X_encoded, y)

    # Save model
    joblib.dump(model, "models/house_price_model.pkl")
    print("Model saved to models/house_price_model.pkl")


if __name__ == "__main__":
    main()
