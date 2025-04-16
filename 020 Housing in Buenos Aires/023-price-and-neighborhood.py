import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")

# PREPARE DATA IMPORT

def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)
    
     # Pull neighborhood
    df["neighborhood"] = df["place_with_parent_names"].str.split("|",expand = True)[3]
    df.drop(columns ="place_with_parent_names",inplace = True)
    

    

    return df


# 2.3.1

# Cell 1: Create a list of filenames for all Buenos Aires real estate CSV files
from glob import glob

files = glob("data/buenos-aires-real-estate-*.csv")


# 2.3.2

# Cell 2: Use the wrangle function to create a list of cleaned DataFrames
import pandas as pd

def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Pull neighborhood
    if 'place_with_parent_names' in df.columns:
        df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
        df.drop(columns="place_with_parent_names", inplace=True)
    
    return df

frames = []
for file in files:
    df = wrangle(file)
    frames.append(df)


# 2.3.3

# Cell 3: Concatenate the items in frames into a single DataFrame
df = pd.concat(frames, ignore_index=True)
print(df.shape)  # Check the shape of the concatenated DataFrame


# 2.3.4

def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Print the column names for debugging
    print("Columns in DataFrame:", df.columns.tolist())

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Pull neighborhood
    if 'place_with_parent_names' in df.columns:
        df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
        df.drop(columns="place_with_parent_names", inplace=True)
    else:
        print("Column 'place_with_parent_names' does not exist in the DataFrame.")

    return df

# 2.3.5

# Cell 4: Create feature matrix X_train and target vector y_train
target = "price_aprox_usd"
features = ["neighborhood"]
y_train = df[target]
X_train = df[features]



# 2.3.6

# Cell 5: Calculate the baseline mean absolute error for your model
from sklearn.metrics import mean_absolute_error

y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
print("Mean apt price:", y_mean)
print("Baseline MAE:", mean_absolute_error(y_train, y_pred_baseline))

# 2.3.7

# Cell 6: Instantiate OneHotEncoder and transform features
from category_encoders import OneHotEncoder

ohe = OneHotEncoder(use_cat_names=True, handle_unknown="ignore")
ohe.fit(X_train)
XT_train = ohe.transform(X_train)
print(XT_train.shape)  # Check the shape of transformed features


# 2.3.8

# Cell 7: Create a pipeline with OneHotEncoder and Ridge regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

model = make_pipeline(
    OneHotEncoder(use_cat_names=True, handle_unknown="ignore"),
    Ridge()
)

model.fit(X_train, y_train)  # Fit the model to training data

# Check your work
check_is_fitted(model[-1])

# 2.3.9
# Cell 8: Predictions for training set and calculate MAE
y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


# 2,3,10

X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()
