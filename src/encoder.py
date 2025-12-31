import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Encoder:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


    # Fit encoder on categorical columns
    # and store the column names for later use during transform step
    # assumes df is preprocessed (i.e missing values handled)
    # and contains only categorical columns to be encoded
    def fit(self, df: pd.DataFrame):
        cat_cols = df.select_dtypes(include="object").columns
        self.encoder.fit(df[cat_cols])
        self.cat_cols = cat_cols

    
    # Transform new data using the fitted encoder
    # and return a dataframe with encoded categorical variables
    # along with numerical columns
    # Assumes df is preprocessed (i.e missing values handled)
    # and contains the same categorical columns as used during fit
    # The output dataframe will have numerical columns unchanged
    # and categorical columns replaced with their one-hot encoded counterparts
    def transform(self, df: pd.DataFrame):
        cat_data = self.encoder.transform(df[self.cat_cols])
        cat_df = pd.DataFrame(cat_data, columns=self.encoder.get_feature_names_out(self.cat_cols))
        num_df = df.drop(self.cat_cols, axis=1)
        return pd.concat([num_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)

    def save(self, path="models/encoder.pkl"):
        joblib.dump(self, path)

    # Load a saved encoder from disk
    @staticmethod
    def load(path="models/encoder.pkl"):
        return joblib.load(path)
