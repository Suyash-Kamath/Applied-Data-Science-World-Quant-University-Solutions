%load_ext autoreload
%autoreload 2

import wqet_grader
from arch.univariate.base import ARCHModelResult

wqet_grader.init("Project 8 Assessment")


# Import your libraries here
import os
import sqlite3
from glob import glob
import joblib
import pandas as pd
import requests
import wqet_grader
from config import settings 
from data import SQLRepository
from IPython.display import VimeoVideo

# 8.5.1

ticker = "MTNOY"  # Corrected ticker symbol
output_size = "full"
data_type = "json"
url = (
    "https://learn-api.wqu.edu/1/data-services/alpha-vantage/query?"
    "function=TIME_SERIES_DAILY&"
    f"symbol={ticker}&"
    f"outputsize={output_size}&"
    f"datatype={data_type}&"
    f"apikey=572f37a53a7b1ffc1333ce0711c18542886ee7a2c24c697978d22dd8e5fb1c7027c9abd72d7eb3656bb9a43e6fed88e719352d1281797210639c3fe3c7481a592ada63af1a3544c96ec85a32408258cf2ec42da685163ba1714b0ffa16b608c2e6bfb4e82f73130250f8aa24632b0f818d1a38767310ce60ca735b7a17ec841e"
)

print("url type:", type(url))
print(url)


# 8.5.2

response = requests.get(url=url)

print("response type:", type(response))


# 8.5.3

response_code = response.status_code
response_data = response.json()
print("code type:", type(response_code))
response_code

# 8.5.4

response_data = response.json()
stock_data = response_data["Time Series (Daily)"]
df_mtnoy = pd.DataFrame.from_dict(stock_data, orient="index", dtype=float)
df_mtnoy.index = pd.to_datetime(df_mtnoy.index)
df_mtnoy.index.name = "date"
df_mtnoy.columns = [c.split(". ") [1] for c in df_mtnoy.columns]
print("df_mtnoy type:", type(df_mtnoy))
df_mtnoy.head()

# 8.5.5

connection = sqlite3.connect(database=settings.db_name,check_same_thread=False)
connection


# 8.5.6

# Reset the index if needed BEFORE inserting into the database
if df_mtnoy.index.name == "date" or isinstance(df_mtnoy.index, pd.DatetimeIndex):
    df_mtnoy = df_mtnoy.reset_index()

# Now insert the updated DataFrame
repo = SQLRepository(connection=connection)
response = repo.insert_table(table_name=ticker, records=df_mtnoy, if_exists="replace")
print("Insert response:", response)

repo = SQLRepository(connection=connection)
response = repo.insert_table(table_name=ticker,records=df_mtnoy,if_exists="replace")
response


df_mtnoy.head()
# Reset the index to make 'date' a column, if it's not already
if df_mtnoy.index.name == "date" or isinstance(df_mtnoy.index, pd.DatetimeIndex):
    df_mtnoy = df_mtnoy.reset_index()

# Confirm 'date' column exists now
print(df_mtnoy.columns)


# 8.5.7

# import pandas as pd

# # Check all tables in the SQLite database
# sql = "SELECT name FROM sqlite_master WHERE type='table';"
# tables = pd.read_sql(sql, con=connection)
# print("Available tables:", tables)

# # The correct table name
# table_name = 'MTNOY'  # The correct table name from the available tables
# if table_name in tables['name'].values:
#     # If the table exists, attempt to read it
#     try:
#         sql = f'SELECT * FROM "{table_name}"'  # Properly quote the table name
#         df_mtnoy_read = pd.read_sql(sql=sql, con=connection, parse_dates=["date"], index_col="date")
#         print("Dataframe head:\n", df_mtnoy_read.head())
#         print("df_mtnoy_read type:", type(df_mtnoy_read))
#         print("df_mtnoy_read shape:", df_mtnoy_read.shape)
#     except Exception as e:
#         print("Error while reading the table:", e)
# else:
#     print(f"Table '{table_name}' does not exist in the database.")


sql = "SELECT * FROM MTNOY"
df_mtnoy_read = pd.read_sql(
    sql=sql,
    con=connection,
    parse_dates=["date"],
    index_col="date"
)

print("df_mtnoy_read type:", type(df_mtnoy_read))
print("df_mtnoy_read shape:", df_mtnoy_read.shape)
df_mtnoy_read.head()


# 8.5.8

def wrangle_data( ticker, n_observations):
    """
    Extract data from the database (or get from AlphaVantage), 
    transform it for training a model, and attach it to self.data.

    Parameters
    ----------
    ticker : str
        The ticker symbol for the equity.

    n_observations : int
        Number of observations to retrieve from the database.

    Returns
    -------
    pd.Series
        The transformed 'return' series for model training.
    """
    
    # Fetch data from the repository (SQL database) based on ticker and limit
    df = repo.read_table(table_name=ticker, limit=n_observations + 1)

    # Check if the DataFrame contains the necessary 'close' column and set 'date' as index
    if 'close' not in df.columns:
        raise ValueError(f"Data for {ticker} does not contain the required 'close' column.")
    
    if 'date' not in df.columns:
        raise ValueError(f"Data for {ticker} does not contain the required 'date' column.")
    
    # Set 'date' as the index if it's not already set
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Sort the data by the index (date) in ascending order
    df.sort_index(ascending=True, inplace=True)

    # Calculate daily returns (percentage change) and add as a new column
    df['return'] = df['close'].pct_change() * 100

    # Drop any missing values (NaN) in the return column
    return df['return'].dropna()


df = repo.read_table(table_name=ticker)
print(df.columns)  # Print columns to see what is available
print(df.head())   # Print first few rows to inspect data



y_mtnoy = wrangle_data(ticker=ticker,n_observations=2500)

print("y_mtnoy type:", type(y_mtnoy))
print("y_mtnoy shape:", y_mtnoy.shape)
y_mtnoy.head()


# 8.5.9

mtnoy_daily_volatility = y_mtnoy.std()


print("mtnoy_daily_volatility type:", type(mtnoy_daily_volatility))
print("MTN Daily Volatility:", mtnoy_daily_volatility)

# 8.5.10

import numpy as np
mtnoy_annual_volatility = mtnoy_daily_volatility*np.sqrt(252)

print("mtnoy_annual_volatility type:", type(mtnoy_annual_volatility))
print("MTN Annual Volatility:", mtnoy_annual_volatility)


# 8.5.11

# Import necessary libraries
import matplotlib.pyplot as plt

# Create fig and ax
fig, ax = plt.subplots(figsize=(15, 6))

# Plot `y_mtnoy` on `ax`
y_mtnoy.plot(ax=ax, label='Daily Return')

# Add axis labels
plt.xlabel('Date')
plt.ylabel('Returns')

# Add title
plt.title("Time Series of MTNOY Returns")

# Add legend
plt.legend()

# Save the plot as a .png file
plt.savefig("images/8-5-11.png", dpi=150)


# 8.5.12

# Import necessary libraries
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Create fig and ax
fig, ax = plt.subplots(figsize=(15, 6))

# Plot ACF of squared returns on `ax`
plot_acf(y_mtnoy**2, ax=ax)

# Add axis labels and title
plt.xlabel("Lag [days]")
plt.ylabel("Correlation coefficient")
plt.title("ACF of MTNOY Squared Returns")

# Save the plot as a .png file
plt.savefig("images/8-5-12.png", dpi=150)


# 8.5.13

from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

# Create fig and ax
fig, ax = plt.subplots(figsize=(15, 6))

# Plot PACF of squared returns on `ax`
plot_pacf(y_mtnoy**2, ax=ax)

# Add axis labels and title
plt.xlabel("Lag [days]")
plt.ylabel("Correlation coefficient")
plt.title("PACF of MTNOY Squared Returns")

# Save the plot as a .png file
plt.savefig("images/8-5-13.png", dpi=150)


# 8.5.14

# Calculate the cutoff index for splitting the data
cutoff_test = int(len(y_mtnoy) * 0.8)

# Split the data into training set (80%) and testing set (20%)
y_mtnoy_train = y_mtnoy[:cutoff_test]

# Display the type and shape of the training data
print("y_mtnoy_train type:", type(y_mtnoy_train))
print("y_mtnoy_train shape:", y_mtnoy_train.shape)

# Show the first few rows of the training data
print(y_mtnoy_train.head())


# 8.5.15

from arch import arch_model

# Build and train model
model = arch_model(y_mtnoy_train, p=1, q=1, rescale=False).fit(disp=0)

# Print the type of the model object
print("model type:", type(model))

# Show model summary
print(model.summary())


# 8.5.16

# Create fig and ax
fig, ax = plt.subplots(figsize=(15, 6))

# Plot standardized residuals on ax
model.std_resid.plot(ax=ax, label='Standardized Residuals')

# Add axis labels
plt.xlabel('Date')
plt.ylabel('Value')
plt.title("MTNOY Garch Model Standardized Residuals")

# Add legend
plt.legend()

# Save the plot to file
plt.savefig("images/8-5-16.png", dpi=150)


# 8.5.17


from statsmodels.graphics.tsaplots import plot_acf

# Create fig and ax
fig, ax = plt.subplots(figsize=(15, 6))

# Plot ACF of squared returns on ax
plot_acf(model.std_resid**2, ax=ax)

# Add axis labels and title
plt.xlabel("Lag [days]")
plt.ylabel("Correlation coefficient")
plt.title("ACF of MTNOY Garch Model Standardized Residuals")

# Save the plot
plt.savefig("images/8-5-17.png", dpi=150)


