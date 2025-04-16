# 2.2. Predicting Price with Location

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wqet_grader
from IPython.display import VimeoVideo
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")

# Prepare Data

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

    

    return df


# Task 2.2.1:** Use your `wrangle` function to create a DataFrame `frame1` from the CSV file `data/buenos-aires-real-estate-1.csv`.


filepath = "data/buenos-aires-real-estate-1.csv"

# Use the wrangle function to process the data
frame1 = wrangle(filepath)

# Display DataFrame information
print(frame1.info())

# Display the first few rows
frame1.head()

# Task 2.2.2: Add to the wrangle function below so that, in the DataFrame it returns, the "lat-lon" column is replaced by separate "lat" and "lon" columns. Don't forget to also drop the "lat-lon" column. Be sure to rerun all the cells above before you continue.

# What's a function?
# Split the strings in one column to create another using pandas.
# Drop a column from a DataFrame using pandas.

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

    # ✅ Split "lat-lon" into "lat" and "lon"
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)

    # ✅ Drop "lat-lon" column
    df = df.drop(columns=["lat-lon"])

    return df
# Define the file path
filepath = "data/buenos-aires-real-estate-1.csv"

# Use the updated wrangle function
frame1 = wrangle(filepath)

# Check DataFrame shape
assert frame1.shape[0] == 1343, f"`frame1` should have 1343 rows, not {frame1.shape[0]}."
assert frame1.shape[1] == 17, f"`frame1` should have 17 columns, not {frame1.shape[1]}."

# Display the first few rows
print(frame1.info())
frame1.head()

#  Task 2.2.3

# Define the new file path
filepath2 = "data/buenos-aires-real-estate-2.csv"

# Use the wrangle function to process the second dataset
frame2 = wrangle(filepath2)

# Check DataFrame shape
assert (
    frame2.shape[0] == 1315
), f"`frame2` should have 1315 rows, not {frame2.shape[0]}."
assert (
    frame2.shape[1] == 17
), f"`frame2` should have 17 columns, not {frame2.shape[1]}."

# Display DataFrame info and first few rows
print(frame2.info())
frame2.head()

# Check your work
assert (
    frame2.shape[0] == 1315
), f"`frame1` should have 1315 rows, not {frame2.shape[0]}."
assert frame2.shape[1] == 17, f"`frame1` should have 17 columns, not {frame2.shape[1]}."



# Task 2.2.4

import pandas as pd

# Concatenate frame1 and frame2
df = pd.concat([frame1, frame2], ignore_index=True)

# Display DataFrame info
print(df.info())

# Display the first few rows
df.head()

# ✅ Check DataFrame shape
assert df.shape == (2658, 17), f"`df` is the wrong size: {df.shape}"

# Task 2.2.5

import plotly.express as px

# Create a Mapbox scatter plot
fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat="lat",  # Latitude column
    lon="lon",  # Longitude column
    width=600,  # Width of map
    height=600,  # Height of map
    color="price_aprox_usd",  # Color based on price
    hover_data=["price_aprox_usd"],  # Display price when hovering
)

# Use OpenStreetMap style
fig.update_layout(mapbox_style="open-street-map")

# Show the figure
fig.show()


# Task 2.2.6

import plotly.express as px

# Create 3D scatter plot
fig = px.scatter_3d(
    df,
    x="lon",  # Longitude on x-axis
    y="lat",  # Latitude on y-axis
    z="price_aprox_usd",  # Price on z-axis
    labels={"lon": "Longitude", "lat": "Latitude", "price_aprox_usd": "Price (USD)"},
    width=600,
    height=500,
)

# Refine formatting
fig.update_traces(
    marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)

# Display figure
fig.show()

# Task 2.2.7


import pandas as pd

# Load dataset (adjust the filename accordingly)
df = pd.read_csv("data/buenos-aires-real-estate-1.csv")  
df.rename(columns={"longitude": "lon", "latitude": "lat"}, inplace=True)
df.columns = df.columns.str.strip()

features = ["lon", "lat"]
X_train = df[features]


# Task 2.2.8

target = "price_aprox_usd"
y_train = df[target]

# task 2.2.9

y_mean = y_train.mean()


# task 2.2.10
y_pred_baseline = [y_mean] * len(y_train)  # Repeat y_mean for every row

# task 2.2.11

from sklearn.metrics import mean_absolute_error

mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price:", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))


# task 2.2.12

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")  # Fill NaN with column mean

# 2.2.13

imputer.fit(X_train)

# 2/2/14
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

# Ensure imputer is fitted before transforming
check_is_fitted(imputer)

# Transform X_train to fill missing values
XT_train = imputer.transform(X_train)

# Convert back to a DataFrame with original column names
XT_train = pd.DataFrame(XT_train, columns=X_train.columns)

# Debugging: Print shape and check for NaN values
print("XT_train shape:", XT_train.shape)
print("NaNs in XT_train:", np.isnan(XT_train).sum().sum())

# Assertions to validate correctness
assert XT_train.shape == (2658, 2), f"`XT_train` is the wrong shape: {XT_train.shape}"
assert np.isnan(XT_train).sum().sum() == 0, "Your feature matrix still has `NaN` values."




import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

# Ensure imputer is fitted before transforming
check_is_fitted(imputer)

# Transform X_train to fill missing values
XT_train = imputer.transform(X_train)

# Convert back to a DataFrame with original column names
XT_train = pd.DataFrame(XT_train, columns=X_train.columns)

# Debugging: Print shape and check for NaN values
print("XT_train shape:", XT_train.shape)
print("NaNs in XT_train:", np.isnan(XT_train).sum().sum())

# Assertions to validate correctness
assert XT_train.shape == (2658, 2), f"`XT_train` is the wrong shape: {XT_train.shape}"
assert np.isnan(XT_train).sum().sum() == 0, "Your feature matrix still has `NaN` values."

# 2.2.15

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Create pipeline with an imputer and a linear regression model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # Fill missing values with mean
    ("linearregression", LinearRegression())  # Linear regression model
])

# Check if the model is instantiated correctly
assert isinstance(model, Pipeline), "Did you instantiate your model?"
model.fit(X_train, y_train)


# 2.2.16

from sklearn.utils.validation import check_is_fitted

# Fit the pipeline to the training data
model.fit(X_train, y_train)

# Check if the model is fitted
check_is_fitted(model["linearregression"])

# Check your work
check_is_fitted(model["linearregression"])

# 2.2.17

# Generate predictions on training data
y_pred_training = model.predict(X_train)

# Check the shape
assert y_pred_training.shape == (2658,)

# Check your work
assert y_pred_training.shape == (2658,)

# 2.2.18

from sklearn.metrics import mean_absolute_error

# Compute the training mean absolute error
mae_training = mean_absolute_error(y_train, y_pred_training)

# Print the result
print("Training MAE:", round(mae_training, 2))


# 2.2.19
