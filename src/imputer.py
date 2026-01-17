# for handling imputer leakage

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer


class MedianImputer:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.num_cols = None

    def fit(self, df: pd.DataFrame):
        self.num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        self.imputer.fit(df[self.num_cols])

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df[self.num_cols] = self.imputer.transform(df[self.num_cols])
        return df

    def save(self, path="models/imputer.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/imputer.pkl"):
        return joblib.load(path)
