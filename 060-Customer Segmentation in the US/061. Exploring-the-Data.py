import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wqet_grader
from IPython.display import VimeoVideo

wqet_grader.init("Project 6 Assessment")


# 6.1.1.

import pandas as pd

# Read the CSV file into the DataFrame
df = pd.read_csv("data/SCFP2019.csv.gz")

# Check the type and shape of the DataFrame
print("df type:", type(df))
print("df shape:", df.shape)

# Display the first few rows of the DataFrame
df.head()


# 6.1.2

# Create a mask where TURNFEAR == 1
mask = df['TURNFEAR'] == 1

# Apply the mask to the DataFrame to create a subset
df_fear = df[mask]

# Check the type and shape of the new DataFrame
print("df_fear type:", type(df_fear))
print("df_fear shape:", df_fear.shape)

# Display the first few rows of the subset
df_fear.head()


# 6.1.3

# Get the unique values in the "AGECL" column
age_groups = df['AGECL'].unique()

# Display the unique age groups
print("Age Groups:", age_groups)


# 6.1.4

# Define the dictionary mapping numbers to age group names
agecl_dict = {
    1: "Under 35",
    2: "35-44",
    3: "45-54",
    4: "55-64",
    5: "65-74",
    6: "75 or Older",
}

# Replace numeric values in the "AGECL" column with the corresponding group names
age_cl = df['AGECL'].map(agecl_dict)

# Display the type and first few rows of the new Series
print("age_cl type:", type(age_cl))
print("age_cl shape:", age_cl.shape)
print(age_cl.head())



# 6.1.5

# Get the value counts for the age_cl Series
age_cl_value_counts = age_cl.value_counts()

# Create the bar plot using pandas
age_cl_value_counts.plot(kind='bar', color='skyblue', figsize=(8, 6))

# Label the axes and set the title
plt.xlabel("Age Group")
plt.ylabel("Frequency (count)")
plt.title("Credit Fearful: Age Groups")

# Display the plot
plt.show()


# 6.1.6

# Plot histogram of "AGE"
# Plot histogram of "AGE" with 10 bins
df_fear["AGE"].plot(kind='hist', bins=10, color='salmon', edgecolor='black', figsize=(8, 6))

# Label axes and title
plt.xlabel("Age")
plt.ylabel("Frequency (count)")
plt.title("Credit Fearful: Age Distribution")

# Show the plot
plt.show()



# 6.1.7

# Mapping numeric race values to descriptive labels
race_dict = {
    1: "White/Non-Hispanic",
    2: "Black/African-American",
    3: "Hispanic",
    5: "Other",
}

# Replace values and create a Series
race = df_fear["RACE"].replace(race_dict)

# Get normalized value counts (proportions)
race_value_counts = race.value_counts(normalize=True)

# Plot horizontal bar chart
race_value_counts.plot(kind='barh', color='skyblue', edgecolor='black', figsize=(8, 5))

# Label and title
plt.xlim((0, 1))
plt.xlabel("Frequency (%)")
plt.ylabel("Race")
plt.title("Credit Fearful: Racial Groups")

# Show plot
plt.show()



# 6.1.8

# Mapping numeric race values to descriptive labels
race_dict = {
    1: "White/Non-Hispanic",
    2: "Black/African-American",
    3: "Hispanic",
    5: "Other",
}

# Replace values in the full dataset
race = df["RACE"].replace(race_dict)

# Get normalized value counts (proportions)
race_value_counts = race.value_counts(normalize=True)

# Plot horizontal bar chart
race_value_counts.plot(kind='barh', color='lightgreen', edgecolor='black', figsize=(8, 5))

# Label and title
plt.xlim((0, 1))
plt.xlabel("Frequency (%)")
plt.ylabel("Race")
plt.title("SCF Respondents: Racial Groups")

# Show plot
plt.show()



# 6.1.9

# Dictionary to map INCCAT values to readable income categories
inccat_dict = {
    1: "0-20",
    2: "21-39.9",
    3: "40-59.9",
    4: "60-79.9",
    5: "80-89.9",
    6: "90-100",
}

# Replace INCCAT values with actual income categories
inccat = df["INCCAT"].replace(inccat_dict)

# Add a new column to df with the readable income categories
df["INCCAT_LABEL"] = inccat

# Group by TURNFEAR and INCCAT_LABEL and calculate normalized frequencies
df_inccat = (
    df.groupby(["TURNFEAR", "INCCAT_LABEL"])
    .size()
    .groupby(level=0)
    .transform(lambda x: x / x.sum())
    .rename("frequency")
    .reset_index()
)

# Print result
print("df_inccat type:", type(df_inccat))
print("df_inccat shape:", df_inccat.shape)
df_inccat


# 6.1.10

import seaborn as sns
import matplotlib.pyplot as plt

# Define the correct order of income categories
income_order = ["0-20", "21-39.9", "40-59.9", "60-79.9", "80-89.9", "90-100"]

# Plot the side-by-side bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=df_inccat, x="INCCAT_LABEL", y="frequency", hue="TURNFEAR", order=income_order)

# Customize labels and title
plt.xlabel("Income Category")
plt.ylabel("Frequency (%)")
plt.title("Income Distribution: Credit Fearful vs. Non-fearful")

# Show plot
plt.show()



# 6.1.11

# Calculate the correlation coefficient
asset_house_corr = df["ASSET"].corr(df["HOUSES"])

# Display result
print("SCF: Asset Houses Correlation:", asset_house_corr)



# 6.1.12

# Calculate the correlation coefficient in the credit-fearful subset
asset_house_corr = df_fear["ASSET"].corr(df_fear["HOUSES"])

# Display the result
print("Credit Fearful: Asset Houses Correlation:", asset_house_corr)


# 6.1.13

# Select the relevant columns from the full dataset
cols = ["ASSET", "HOUSES", "INCOME", "DEBT", "EDUC"]
corr = df[cols].corr()

# Display the correlation matrix with a color gradient
corr.style.background_gradient(cmap='coolwarm', axis=None)


# 6.1.14

