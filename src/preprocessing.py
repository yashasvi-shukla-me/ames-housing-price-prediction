import pandas as pd
import numpy as np


# The columns with missing values that indicate absence of a feature
# e.g. no garage, no basement, etc.
# These column do not contain numerical values and are categorical in nature
# and should be filled with 'None' string for missing values.
# For missing values the dataset has NaN which needs to be replaced with 'None'

none_cols = [
    'PoolQC','Alley','Fence','FireplaceQu',
    'GarageType','GarageFinish','GarageQual','GarageCond',
    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'MasVnrType'
]

# These column contain numerical values and missing values should be filled
# with the median of the respective column.

num_cols = ['LotFrontage','MasVnrArea','GarageYrBlt']


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill categorical absent features with 'None'
    df[none_cols] = df[none_cols].fillna('None')

    # Fill numeric missing values
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df




def prepare_features(df: pd.DataFrame, is_train: bool = True):

    df = clean_data(df)

    # If training data, apply log transformation to SalePrice
    # to reduce skewness
    if is_train:
        df['SalePrice'] = np.log1p(df['SalePrice'])

    # Convert categorical variables to dummy/indicator variables
    df = pd.get_dummies(df)

    return df
