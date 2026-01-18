from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.inference import predict_price # using absolute import

app = FastAPI(title="Ames House Price Prediction API")


# This defines the JSON schema my API accepts
class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    YearBuilt: int


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: HouseInput):
    # Convert JSON -> DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict
    price = predict_price(df)

    return {
        "predicted_price": round(price, 2)
    }