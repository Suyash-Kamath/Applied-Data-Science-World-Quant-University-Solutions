import warnings

import wqet_grader

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")


import warnings
from glob import glob

import pandas as pd
import seaborn as sns
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt


# prepare data : wrangle function

def wrangle(filepath):
    df = pd.read_csv(filepath)
    mask1 = (df['property_type'] == 'apartment')
    mask2 = (df['price_aprox_usd'] < 100000)
    mask3 = (df['place_with_parent_names'].str.contains('Distrito Federal'))


    df = df[mask1 & mask2 & mask3]

    low, high = df['surface_covered_in_m2'].quantile([0.1, 0.9])
    maskArea = df['surface_covered_in_m2'].between(low, high)
    df = df[maskArea]

    df[['lat', 'lon']] = df['lat-lon'].str.split(',', expand=True).astype(float)
    df = df.drop(columns='lat-lon')

    df['borough'] = df['place_with_parent_names'].str.split('|', expand=True)[1]
    df = df.drop(columns='place_with_parent_names')

    columns_na = [i for i in df.columns if df[i].isna().sum() > len(df) // 2]
    df = df.drop(columns = columns_na)

    list_card = ["operation", "property_type",  "currency", "properati_url"]
    df = df.drop(columns = list_card)

    #leakage 
    highlow_cardinality = ['price', 'price_aprox_local_currency', 'price_per_m2']
    df = df.drop(columns = highlow_cardinality)



    return df


df.info()

frame = []
for i in files:
    df = wrangle(i)
    frame.append(df)


# In[75]:


df = pd.concat(frame)
df.info()


# 2.5.2

files = glob('data/mexico-city-real-estate-[0-5].csv')
files


# 2.5.3

frame = []
for i in files:
    df = wrangle(i)
    frame.append(df)


# In[80]:


df = pd.concat(frame)
print(df.info())
df.head()


# In[81]:


df.shape

# 2.5.4

# Plot distribution of price
plt.hist(df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel('Count')
plt.title("Distribution of Apartment Sizes")

# Don't delete the code below 👇
plt.savefig("images/2-5-4.png", dpi=150)



# 2.5.5

plt.scatter(x = df['surface_covered_in_m2'], y=df['price_aprox_usd'], )
plt.ylabel('Price [USD]')
plt.xlabel('Area [sq meters]')
plt.title('Mexico City: Price vs. Area')
# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)


# 2.5.6

# pip install plotly
import plotly.express as px
import pandas as pd

# Assuming df is your dataframe
# Example DataFrame (ensure you have lat, lon, and price_aprox_usd columns)

# Create the mapbox scatter plot
fig = px.scatter_mapbox(df,
                        lat='lat',               # Latitude column
                        lon='lon',               # Longitude column
                        color='price_aprox_usd', # Color based on price
                        hover_name='borough',    # Show borough when hovered over
                        hover_data={'price_aprox_usd': True, 'surface_covered_in_m2': True},  # Additional info on hover
                        title='Real Estate Prices in Mexico City',
                        color_continuous_scale='Viridis', # Color scale for price
                        size_max=10,             # Adjust marker size
                        zoom=10,                 # Zoom level of the map
                        height=600)

# Set Mapbox access token (replace with your Mapbox token)
fig.update_layout(mapbox_style="open-street-map")  # Alternatively, you can use 'carto-positron' or other Mapbox styles
fig.show()


# 2.5.7

# Split data into feature matrix `X_train` and target vector `y_train`.
X_train = df[['surface_covered_in_m2', 'lat', 'lon', 'borough']]
y_train = df[feature]

# 2.5.8

y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

# 2.5.9

model = make_pipeline(
    OneHotEncoder(use_cat_names = True), 
    SimpleImputer(),
    Ridge()
)
model.fit(X_train, y_train)


# 2.5.10


X_test = pd.read_csv('./data/mexico-city-test-features.csv')
print(X_test.info())
X_test.head()

# 2.5.11

import pandas as pd

# Convert predictions to a pandas Series
y_test_pred = pd.Series(y_test_pred)

# 2.5.12

# Extract coefficients from the Ridge model
coefficients = model.named_steps['ridge'].coef_

# Extract feature names from the OneHotEncoder step
feature_names = model.named_steps['onehotencoder'].get_feature_names_out()

# Create the feature importance Series
feat_imp = pd.Series(coefficients, index=feature_names)

# Sort by absolute value
feat_imp = feat_imp.reindex(feat_imp.abs().sort_values().index)

# Display the result
print(feat_imp)


# 2.5.13



# Create horizontal bar chart
plt.barh(feature_names,feat_imp)
plt.title('Feature Importances for Apartment Price')
plt.ylabel('Feature')
plt.xlabel('Importance [USD]')
plt.show()
# Don't delete the code below 👇
plt.savefig("images/2-5-13.png", dpi=150)

