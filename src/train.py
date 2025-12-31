import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import prepare_features


def main():
    # Load the training data
    train = pd.read_csv("data/raw/train.csv")
    
    # Prepare features
    data = prepare_features(train, is_train=True)

    # Split X and y
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    # Train the model
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    # Validate model
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse = (-scores.mean()) ** 0.5
    print("Cross-validated RMSE:", rmse)

    # Fit on full data
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, "models/house_price_model.pkl")
    print("Model saved to models/house_price_model.pkl")


if __name__ == "__main__":
    main()
