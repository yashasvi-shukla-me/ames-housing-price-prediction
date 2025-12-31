import pandas as pd
from preprocessing import clean_data, prepare_features

# simple test to verify the preprocessing functions

# Load the dataset
train = pd.read_csv("data/raw/train.csv")


# we will get the cleaned data
# i.e missing values handled
# for numerical columns filled with median
# for categorical columns filled with 'None'
cleaned = clean_data(train)
print(cleaned.head())


# now prepare features
# this will also apply log transformation to SalePrice
# and convert categorical variables to dummies
prepared = prepare_features(train, is_train=True)
print(prepared.shape)
print(prepared.head())