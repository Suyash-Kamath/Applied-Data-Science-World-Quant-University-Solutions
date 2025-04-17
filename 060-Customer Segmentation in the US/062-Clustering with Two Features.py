
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wqet_grader
from IPython.display import VimeoVideo
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted
from teaching_tools.widgets import ClusterWidget, SCFClusterWidget

wqet_grader.init("Project 6 Assessment")


# 6.2.1

import pandas as pd

def wrangle(filepath):
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)
    
    # Step 2: Filter the DataFrame to only include rows where TURNFEAR is 1
    # Assuming TURNFEAR column indicates whether the household has been turned down or feared being denied credit
    df_filtered = df[df['TURNFEAR'] == 1]
    
    # Step 3: Return the filtered DataFrame
    return df_filtered

# Example usage:
# df = wrangle('path_to_your_file.csv')
# print(df.head())


# 6.2.2

# Assuming the wrangle function is already defined as shown previously

# Task 6.2.2: Read the file SCFP2019.csv.gz into a DataFrame
df = wrangle('data/SCFP2019.csv.gz')


# Print the type and shape of the DataFrame
print("df type:", type(df))
print("df shape:", df.shape)

# Display the first few rows of the DataFrame
df.head()


# 6.2.3

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing 'HOUSES' and 'DEBT' columns
plt.figure(figsize=(10, 6))

# Create the scatter plot
sns.scatterplot(data=df, x='DEBT', y='HOUSES')

# Label the axes and title
plt.xlabel("Household Debt [$1M]")
plt.ylabel("Home Value [$1M]")
plt.title("Credit Fearful: Home Value vs. Household Debt")

# Show the plot
plt.show()



# 6.2.4

# Create the feature matrix X containing "DEBT" and "HOUSES"
X = df[['DEBT', 'HOUSES']]

# Check the type, shape, and first few rows of X
print("X type:", type(X))
print("X shape:", X.shape)
X.head()


# 6.2.5

cw = ClusterWidget(n_clusters=3)
cw.show()


# 6.2.6

scfc = SCFClusterWidget(x=df["DEBT"], y=df["HOUSES"], n_clusters=3)
scfc.show()


# 6.2.7

from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

# Build the model with 3 clusters and set the random state for reproducibility
model = KMeans(n_clusters=3, random_state=42)

# Print the model type to check if the correct model is built
print("model type:", type(model))

# Fit the model to the data (X is the feature matrix)
model.fit(X)

# Assert that the model has been fitted to the data
check_is_fitted(model)



# 6.2.8

# Extract the labels assigned to each data point during training
labels = model.labels_

# Print the type, shape, and first 10 labels
print("labels type:", type(labels))
print("labels shape:", labels.shape)
print("First 10 labels:", labels[:10])



# 6.2.9

import seaborn as sns
import matplotlib.pyplot as plt

# Create a new DataFrame that includes "DEBT", "HOUSES", and "labels"
df_plot = X.copy()
df_plot['labels'] = labels

# Create the scatter plot using seaborn
sns.scatterplot(data=df_plot, x="DEBT", y="HOUSES", hue="labels", palette="deep")

# Add labels and title
plt.xlabel("Household Debt [$1M]")
plt.ylabel("Home Value [$1M]")
plt.title("Credit Fearful: Home Value vs. Household Debt")

# Show the plot
plt.show()



# 6.2.10

# Extract centroids
centroids = model.cluster_centers_

# Check the type and shape of centroids
print("centroids type:", type(centroids))
print("centroids shape:", centroids.shape)

# Display the centroids
centroids



# 6.2.11

# Plot "HOUSES" vs "DEBT" using seaborn
sns.scatterplot(data=df, x='DEBT', y='HOUSES', hue=labels, palette='deep')

# Add centroids to the plot
plt.scatter(centroids[:, 0], centroids[:, 1], c='gray', marker='X', s=200, label='Centroids')

# Label the axes and title
plt.xlabel("Household Debt [$1M]")
plt.ylabel("Home Value [$1M]")
plt.title("Credit Fearful: Home Value vs. Household Debt")

# Display the plot
plt.legend()
plt.show()

# 6.2.12

# Extract the inertia
inertia = model.inertia_

# Display the inertia
print("inertia type:", type(inertia))
print("Inertia (3 clusters):", inertia)


# 6.2.13

from sklearn.metrics import silhouette_score

# Calculate the silhouette score
ss = silhouette_score(X, labels)

# Display the silhouette score
print("ss type:", type(ss))
print("Silhouette Score (3 clusters):", ss)



# 6.2.14
# Define the range of cluster values to try
n_clusters = list(range(2, 11))

# Initialize empty lists to store inertia and silhouette scores
inertia_errors = []
silhouette_scores = []

# Loop through each cluster count
for k in n_clusters:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    
    # Append inertia
    inertia_errors.append(model.inertia_)
    
    # Append silhouette score
    score = silhouette_score(X, model.labels_)
    silhouette_scores.append(score)

# Display the results
print("inertia_errors type:", type(inertia_errors))
print("inertia_errors len:", len(inertia_errors))
print("Inertia:", inertia_errors)
print()
print("silhouette_scores type:", type(silhouette_scores))
print("silhouette_scores len:", len(silhouette_scores))
print("Silhouette Scores:", silhouette_scores)


# 6.2.15

# Plot `inertia_errors` by `n_clusters`
import matplotlib.pyplot as plt

# Define the range for n_clusters (2 to 12)
n_clusters_range = range(2, 13)

# Create a line plot
plt.plot(n_clusters_range, inertia_errors, marker='o')

# Label the axes and title
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("K-Means Model: Inertia vs Number of Clusters")

# Show the plot
plt.show()



# 6.2.16

# Plot `silhouette_scores` vs `n_clusters`
import matplotlib.pyplot as plt

# Define the range for n_clusters (2 to 12)
n_clusters_range = range(2, 13)

# Create a line plot
plt.plot(n_clusters_range, silhouette_scores, marker='o', color='b')

# Label the axes and title
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("K-Means Model: Silhouette Score vs Number of Clusters")

# Show the plot
plt.show()



# 6.2.17

from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

# Build the final model with the optimal number of clusters (e.g., 4)
final_model = KMeans(n_clusters=4, random_state=42)

# Fit the model to the data
final_model.fit(X)

# Check if the model has been fitted
check_is_fitted(final_model)

print("final_model type:", type(final_model))
