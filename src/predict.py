import pandas as pd
from src.inference import predict_price # using absolute import


def main():
    df = pd.read_csv("data/raw/train.csv")

    sample = df.iloc[[0]].copy()
    actual_price = sample["SalePrice"].values[0]

    sample = sample.drop("SalePrice", axis=1)

    predicted_price = predict_price(sample)

    print("Actual price:", actual_price)
    print("Predicted price:", round(predicted_price, 2))


if __name__ == "__main__":
    main()