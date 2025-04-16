import warnings

import wqet_grader

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 3 Assessment")



import inspect
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg


# In[4]:


from pprint import PrettyPrinter



# 3.5.1

client = MongoClient(host="localhost", port=27017)
db = client["air-quality"]
dar = db["dar-es-salaam"]



# 3.5.2

sites = dar.distinct("metadata.site")
sites

# 3.5.3

result = dar.find({"metadata.measurement": "P2.5"}).limit(5) 



# In[15]:


readings_per_site = [{'_id':23, 'count': dar.count_documents({"metadata.site": 23})},
         {'_id':11, 'count': dar.count_documents({"metadata.site": 11})}]
readings_per_site

# 3.5.4

def wrangle(collection):

    results = collection.find(
        {"metadata.site": 11, "metadata.measurement": "P2"}, ## 11 has the most count
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read results into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Dar_es_Salaam")

    # Remove outliers
    df = df[df["P2"] < 100]

    return df["P2"].resample("1H").mean().fillna(method="ffill")


# Use your `wrangle` function to query the `dar` collection and return your cleaned results.

# In[19]:
y = wrangle(dar)
y.head()


# 3.5.5

fig, ax = plt.subplots(figsize=(15, 6))

y.plot(  #used y instead of y["P2"] since our y is a pd Series
    xlabel="Date",
    ylabel="PM2.5 Level",
    title="Dar es Salaam PM2.5 Levels",
)


# Don't delete the code below 👇
plt.savefig("images/3-5-5.png", dpi=150)


# 3.5.6

fig, ax = plt.subplots(figsize=(15, 6))

# Don't delete the code below 👇
plt.savefig("images/3-5-6.png", dpi=150)


# 3.5.7


fig, ax = plt.subplots(figsize=(15, 6))

plot_acf(y, ax=ax)
plt.xlabel= "Lag [hours]"
plt.ylabel = "Correlation Coefficient"
plt.title = "Dar es Salaam PM2.5 Readings, ACF"

# Don't delete the code below 👇
plt.savefig("images/3-5-7.png", dpi=150)




# 3.5.8

fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
plt.xlabel= "Lag [hours]"
plt.ylabel = "Correlation Coefficient"
plt.title = "Dar es Salaam PM2.5 Readings, PACF"

# Don't delete the code below 👇
plt.savefig("images/3-5-8.png", dpi=150)



# 3.5.9

cutoff_test = int(len(y)*.9)

y_train = y[:cutoff_test]
y_test = y[cutoff_test:]

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 3.5.10

y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", y_train_mean)
print("Baseline MAE:", mae_baseline)



# 3.5.11


from statsmodels.tsa.ar_model import AutoReg


# In[41]:


#AR autoregressive
p_params = range(1, 31)
maes = []
order = ""
for p in p_params:

    # Note start time
    start_time = time.time()
    # Train model
    model = AutoReg(y_train, lags=p, old_names=True).fit()
    # Calculate model training time
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Trained AutoReg {order} in {elapsed_time} seconds.")
    # Generate in-sample (training) predictions
    y_pred = model.predict().dropna()
    # Calculate training MAE
    mae = mean_absolute_error(y_train.loc[y_pred.index], y_pred)
    # Append MAE to list in dictionary
    maes.append(mae)

mae_series = pd.Series(maes, name="mae", index=p_params)
mae_series.head()


# In[42]:


mae_series = mae_series.sort_values()






# 3.5.12


best_p = 28
best_model = AutoReg(y_train, lags=28, old_names=True).fit()


# In[45]:


type(best_model)

# 3.5.13

y_pred = best_model.predict().dropna()


# In[48]:


y_train_resid = best_model.resid
y_train_resid.name = "residuals"
y_train_resid.head()


# In[49]:


y_train_resid = y_train.loc[y_pred.index] - y_pred
y_train_resid.name = "residuals"


# 3.5.14

y_train_resid.hist()
plt.xlabel = "Residuals"
plt.ylabel = "Frequency"
plt.title = "Best Model, Training Residuals"

# Don't delete the code below 👇
plt.savefig("images/3-5-14.png", dpi=150)



# 3.5.15

fig, ax = plt.subplots(figsize=(15, 6))

# Don't delete the code below 👇
plt.savefig("images/3-5-15.png", dpi=150)






# 3.5.16

y_pred_wfv = pd.Series(dtype=float)  # Specify dtype explicitly
y_pred_wfv = pd.concat([y_pred_wfv, next_pred])
history = history.append(pd.Series(y_test[next_pred.index[0]]))
history = history.append(pd.Series(y_test[next_pred.index[0]]))
# Perform walk-forward validation
y_pred_wfv = pd.Series(dtype=float)  # Initialize empty Series with explicit dtype
history = y_train.copy()  # Start with training data

for timestamp in y_test.index:  # Iterate over test timestamps
    # Train model on history
    model = AutoReg(history, lags=28).fit()
    
    # Forecast next value
    next_pred = model.forecast()
    
    # Append prediction to y_pred_wfv
    y_pred_wfv = pd.concat([y_pred_wfv, pd.Series(next_pred, index=[timestamp])])
    
    # Update history with actual test value
    history = history.append(pd.Series(y_test[timestamp], index=[timestamp]))

# Set Series metadata
y_pred_wfv.name = "prediction"
y_pred_wfv.index.name = "timestamp"

# Display results
print(y_pred_wfv.head())




# 3.5.17 is 
wqet_grader.grade("Project 3 Assessment", "Task 3.5.17", y_pred_wfv)


# 3.5.18


df_pred_test = pd.DataFrame(
    {
        "y_test": y_test, "y_pred_wfv": y_pred_wfv
    }
)
fig = px.line(df_pred_test, labels={"value": "PM2.5"})
fig.update_layout(
    title="Dar es Salaam, WFV Predictions",
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
)
# Don't delete the code below 👇
fig.write_image("images/3-5-18.png", scale=1, height=500, width=700)

fig.show()





