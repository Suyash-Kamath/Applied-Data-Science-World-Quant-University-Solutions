import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy
import wqet_grader
from IPython.display import VimeoVideo
from pymongo import MongoClient
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.power import GofChisquarePower
from teaching_tools.ab_test.experiment import Experiment
from teaching_tools.ab_test.reset import Reset

wqet_grader.init("Project 7 Assessment")


# Reset database
r = Reset()
r.reset_database()

# 7.3.1

from pymongo import MongoClient

# Step 1: Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string as needed

# Step 2: Access the database
db = client["wqu-abtest"]

# Step 3: Access the collection
ds_app = db["ds-applicants"]

# Check the types
print("client:", type(client))
print("ds_app:", type(ds_app))


# 7.3.2

chi_square_power = GofChisquarePower()
group_size = math.ceil(
chi_square_power.solve_power(effect_size=0.2, alpha=0.05, power=0.8) )
print("Group size:", group_size)
print("Total # of applicants needed:", group_size * 2)

# 7.3.3
n_observations = np.arange(0, group_size * 2 + 1)  # Correcting the formula for n_observations
effect_sizes = np.array([0.2, 0.5, 0.8])

# Plot power curve using 'chi_square_power'
chi_square_power.plot_power(
    dep_var="nobs",
    nobs=n_observations,
    effect_size=effect_sizes,
    alpha=0.05,
    n_bins=2
)


# 7.3.4

# Perform the aggregation and convert the result to a list
result = list(ds_app.aggregate([
    {"$match": {"admissionsQuiz": "incomplete"}},
    {
        "$group": {
            "_id": {"$dateTrunc": {"date": "$createdAt", "unit": "day"}},
            "count": {"$sum": 1}
        }
    }
]))

# Print the type of result to confirm it's a list
print("result type:", type(result))

# Print the first few documents in the result to inspect
print(result[:5])


# 7.3.5

no_quiz = (
    pd.DataFrame(result)
    .rename({"_id": "date", "count": "new_users"}, axis=1)
    .set_index("date")
    .sort_index()
    .squeeze()
)

# Print type and shape of no_quiz
print("no_quiz type:", type(no_quiz))
print("no_quiz shape:", no_quiz.shape)

# Display the first few rows of no_quiz
no_quiz.head()


# 7.3.6

# Create histogram of `no_quiz
no_quiz.hist()
# Add axis labels and title
plt.xlabel("No-Quiz Applicants")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Daily No-Quiz Applicants")

# 7.3.7

mean = no_quiz.describe()["mean"]
std = no_quiz.describe()["std"]
print("no_quiz mean:", mean)
print("no_quiz std:", std)


# 7.3.8

days = 10
sum_mean = mean*days
sum_std = std*np.sqrt(days)
print("Mean of sum:", sum_mean)
print("Std of sum:", sum_std)

# 7.3.9

prob_400_or_fewer = scipy.stats.norm.cdf(group_size*2,loc=sum_mean,scale=sum_std)
prob_400_or_greater = 1-prob_400_or_fewer

print(
    f"Probability of getting 400+ no_quiz in {days} days:",
    round(prob_400_or_greater, 3),
)

# 7.3.10

exp = Experiment(repo=client, db="wqu-abtest", collection="ds-applicants")
exp.reset_experiment()
result = exp.run_experiment(days=days)
print("result type:", type(result))
result

# 7.3.11

result = ds_app.find({"inExperiment":True})
print("results type:", type(result))


# 7.3.12

df = pd.DataFrame(result).dropna()

print("df type:", type(df))
print("df shape:", df.shape)
df.head()



# 7.3.13

# Create a 2x2 contingency table of 'group' vs 'admitted'
data = pd.crosstab(index=df["group"], columns=df["admitted"])

print("data type:", type(data))
print("data shape:", data.shape)
data
