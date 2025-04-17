import wqet_grader
from pymongo import MongoClient
from pymongo.collection import Collection
from teaching_tools.ab_test.reset import Reset

wqet_grader.init("Project 7 Assessment")

r = Reset()
r.reset_database()

# Import your libraries here
# Import your libraries here
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.power import GofChisquarePower
from teaching_tools.ab_test.experiment import Experiment
from country_converter import CountryConverter
from pymongo.collection import Collection
from pymongo import MongoClient
from pprint import PrettyPrinter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import scipy
import plotly.express as px

# 7.5.1

# Import the MongoClient class
from pymongo import MongoClient

# Create a Mongo client
client = MongoClient(host="localhost", port=27017)

# Create or connect to the database
db = client["wqu-abtest"]

# Access the collection "mscfe-applicants"
mscfe_app = db["mscfe-applicants"]


# 7.5.2

# Aggregate applicants by nationality
result = mscfe_app.aggregate([
    {
        "$group": {
            "_id": "$countryISO2",
            "count": {"$sum": 1}
        }
    }
])

# Load result into DataFrame
df_nationality = pd.DataFrame(result).rename(
    columns={"_id": "country_iso2"}
).sort_values("count")

print("df_nationality type:", type(df_nationality))
print("df_nationality shape:", df_nationality.shape)
df_nationality.head()


# 7.5.3

from country_converter import CountryConverter

# Instantiate CountryConverter
cc = CountryConverter()

# Create new columns for full country names and ISO3 codes
df_nationality["country_name"] = cc.convert(
    df_nationality["country_iso2"], to="name_short"
)

df_nationality["country_iso3"] = cc.convert(
    df_nationality["country_iso2"], to="ISO3"
)

print("df_nationality type:", type(df_nationality))
print("df_nationality shape:", df_nationality.shape)
df_nationality.head()


# 7.5.4

import plotly.express as px

# Create new column: percentage of applicants per country
df_nationality["count_pct"] = (df_nationality["count"] / df_nationality["count"].sum()) * 100

# Define function to build choropleth
def build_nat_choropleth():
    fig = px.choropleth(
        data_frame=df_nationality,
        locations="country_iso3",
        color="count_pct",
        projection="natural earth",
        color_continuous_scale=px.colors.sequential.Oranges,
        title="MScFE Applicants: Nationalities"
    )
    return fig

# Generate the figure
nat_fig = build_nat_choropleth()

# Save and display the figure
nat_fig.write_image("images/7-5-4.png", scale=1, height=500, width=700)
nat_fig.show()


# ETL

# class MongoRepository:
#     """Repository class for interacting with MongoDB database.

#     Parameters
#     ----------
#     client : `pymongo.MongoClient`
#         By default, `MongoClient(host='localhost', port=27017)`.
#     db : str
#         By default, `'wqu-abtest'`.
#     collection : str
#         By default, `'mscfe-applicants'`.

#     Attributes
#     ----------
#     collection : pymongo.collection.Collection
#         All data will be extracted from and loaded to this collection.
#     """

#     # Task 7.5.5: `__init__` method
    

#     # Task 7.5.6: `find_by_date` method
    

#     # Task 7.5.7: `update_applicants` method
    

#     # Task 7.5.7: `assign_to_groups` method
    

#     # Task 7.5.14: `find_exp_observations` method
from pymongo import MongoClient
import pandas as pd
import random

class MongoRepository:
    """Repository class for interacting with MongoDB database.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        By default, `MongoClient(host='localhost', port=27017)`.
    db : str
        By default, `'wqu-abtest'`.
    collection : str
        By default, `'mscfe-applicants'`.

    Attributes
    ----------
    collection : pymongo.collection.Collection
        All data will be extracted from and loaded to this collection.
    """

    # Task 7.5.5: `__init__` method
    def __init__(
        self,
        client=MongoClient(host="localhost", port=27017),
        db="wqu-abtest",
        collection="mscfe-applicants"
    ):
        self.collection = client[db][collection]

    # Task 7.5.6: `find_by_date` method
    def find_by_date(self, date_string):
        start = pd.to_datetime(date_string)
        end = start + pd.DateOffset(days=1)

        query = {
            "createdAt": {"$gte": start, "$lt": end},
            "admissionsQuiz": "incomplete"
        }

        result = self.collection.find(query)
        observations = list(result)
        return observations

    # Task 7.5.7: `update_applicants` method
    def update_applicants(self, observations_assigned):
        n = 0
        n_modified = 0

        for doc in observations_assigned:
            result = self.collection.update_one(
                filter={"_id": doc["_id"]},
                update={"$set": doc}
            )
            n += result.matched_count
            n_modified += result.modified_count

        transaction_result = {"n": n, "nModified": n_modified}
        return transaction_result

    # Task 7.5.7: `assign_to_groups` method
    def assign_to_groups(self, date_string):
        observations = self.find_by_date(date_string)

        random.seed(42)
        random.shuffle(observations)

        idx = len(observations) // 2

        for doc in observations[:idx]:
            doc["inExperiment"] = True
            doc["group"] = "no email (control)"

        for doc in observations[idx:]:
            doc["inExperiment"] = True
            doc["group"] = "email (treatment)"

        result = self.update_applicants(observations)
        return result

    # Task 7.5.14: `find_exp_observations` method
    def find_exp_observations(self):
        result = self.collection.find({"inExperiment": True})
        return list(result)

# 7.5.5.

repo = MongoRepository()
print("repo type:", type(repo))
repo

# 7.5.8

from statsmodels.stats.power import GofChisquarePower
import math

# Instantiate the Chi-Square Power calculator
chi_square_power = GofChisquarePower()

# Calculate required group size for desired power and effect size
group_size = math.ceil(chi_square_power.solve_power(
    effect_size=0.5,   # medium effect: 0.5 (small: 0.2, large: 0.8)
    alpha=0.05,        # significance level
    power=0.8          # desired power of the test
))

# Output results
print("Group size:", group_size)
print("Total # of applicants needed:", group_size * 2)


# 7.5.9

# Aggregate no-quiz applicants by sign-up date
result = mscfe_app.aggregate([
    {
        "$match": {"admissionsQuiz": "incomplete"}
    },
    {
        "$group": {
            "_id": {
                "$dateTrunc": {
                    "date": "$createdAt",
                    "unit": "day"
                }
            },
            "count": {"$sum": 1}
        }
    }
])

# Load result into DataFrame
no_quiz_mscfe = (
    pd.DataFrame(result)
    .rename(columns={"_id": "date", "count": "new_users"})
    .set_index("date")
    .sort_index()
    .squeeze()
)

# Display information
print("no_quiz type:", type(no_quiz_mscfe))
print("no_quiz shape:", no_quiz_mscfe.shape)
no_quiz_mscfe.head()


# 7.5.10

mean = no_quiz_mscfe.describe()["mean"]
std = no_quiz_mscfe.describe()["std"]
print("no_quiz mean:", mean)
print("no_quiz std:", std)

# Ungrade Task

exp_days = 7
sum_mean = mean*exp_days
sum_std = std*np.sqrt(exp_days)
print("Mean of sum:", sum_mean)
print("Std of sum:", sum_std)

# 7.5.11

prob_65_or_fewer = scipy.stats.norm.cdf(
    group_size * 2,
    loc=sum_mean,
    scale=sum_std
)

prob_65_or_greater = 1 - prob_65_or_fewer  # Corrected this line
print(
    f"Probability of getting 65+ no_quiz in {exp_days} days:",
    round(prob_65_or_greater, 3),
)

# 7.5.12

exp = Experiment(repo=client, db="wqu-abtest", collection="mscfe-applicants")
exp.reset_experiment()
result = exp.run_experiment(days=exp_days, assignment=True)
print("result type:", type(result))
result

# 7.5.14

result = repo.find_exp_observations()
df = pd.DataFrame (result).dropna()
print("df type:", type(df))
print("df shape:", df.shape)
df.head()

# 7.5.15

data= pd.crosstab(
index=df["group"],
columns=df["admissionsQuiz"],
normalize=False
)
print("data type:", type(data))
print("data shape:", data.shape)
data

# 7.5.16

def build_contingency_bar():
    # Create side-by-side bar chart
    fig = px.bar(
        data_frame=data,
        barmode="group",
        title="MScFE: Admissions Quiz Completion by Group"
    )
    # Set axis labels
    fig.update_layout(
        xaxis_title="Group", 
        yaxis_title="Frequency [count]"
    )
    return fig

# Don't delete the code below
cb_fig = build_contingency_bar()
cb_fig.write_image("images/7-5-16.png", scale=1, height=500, width=700)
cb_fig.show()


# 7.5.17

contingency_table = Table2x2(data.values)

print("contingency_table type:", type(contingency_table))
contingency_table.table_orig

# 7.5.18


# Assuming contingency_table is a valid object that has the method test_nominal_association
# For example, if using `association` from the `association` package (or similar)

# Perform chi-square test for nominal association
chi_square_test = contingency_table.test_nominal_association()

# Print the type of the chi-square test result
print("chi_square_test type:", type(chi_square_test))

# Print the chi-square test result
print(chi_square_test)


# 7.5.19

odds_ratio = contingency_table.oddsratio.round(1)
print("Odds ratio:", odds_ratio)
