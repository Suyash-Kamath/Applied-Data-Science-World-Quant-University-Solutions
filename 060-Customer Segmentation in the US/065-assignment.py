import wqet_grader

wqet_grader.init("Project 6 Assessment")


# Import libraries here
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from teaching_tools.widgets import ClusterWidget, SCFClusterWidget
from scipy.stats.mstats import trimmed_var
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 6.5.1

df = pd.read_csv("data/SCFP2019.csv.gz")
print("df shape:", df.shape)
df.head()

# 6.5.2

prop_biz_owners = sum(df["HBUS"]) / (sum(df["HBUS"] == 0) + sum(df["HBUS"])) 
print("% of business owners in df:", prop_biz_owners)

# 6.5.3

inccat_dict = {
    1: "0-20",
    2: "21-39.9",
    3: "40-59.9",
    4: "60-79.9",   # Fixed missing key
    5: "80-89.9",
    6: "90-100",
}

df_inccat = (
    df["INCCAT"]
    .replace(inccat_dict)
    .groupby(df["HBUS"])
    .value_counts(normalize=True)
    .rename("frequency")
    .to_frame()
    .reset_index()
)

df_inccat



# 6.5.4

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(
    x="INCCAT",
    y="frequency",
    hue="HBUS",
    data=df_inccat,
    order=list(inccat_dict.values())  # Ensure order is a list
)

plt.xlabel("Income Category")  # Replace with your actual x-axis title
plt.ylabel("Proportion of Households")  # Replace with your actual y-axis title
plt.title("Distribution of Income Categories by HBUS")  # Replace with your actual chart title

# Don't delete the code below
plt.savefig("images/6-5-4.png", dpi=150)




# 6.5.5

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    x=df["DEBT"] / 1e6,
    y=df["HOUSES"] / 1e6,
    palette="deep"
)

plt.xlabel("Household Debt")
plt.ylabel("Home Value")
plt.title("Home Value vs. Household Debt")

# Don't delete the code below
plt.savefig("images/6-5-5.png", dpi=150)



# 6.5.6



# Filter the DataFrame for small businesses based on HBUS and INCOME
mask = (df["HBUS"]) & (df["INCOME"] < 500_000)
df_small_biz = df[mask]

# Display the shape and first few rows
print("df_small_biz shape:", df_small_biz.shape)
df_small_biz.head()



# 6.5.7


df_small_biz["AGE"].hist()
# Don't delete the code below 👇
plt.savefig("images/6-5-7.png", dpi=150)


# 6.5.8

# Calculate variance, get 10 largest features
top_ten_var = df_small_biz.var().sort_values().tail(10)
top_ten_var


# 6.5.9


# Calculate trimmed variance
top_ten_trim_var = df_small_biz.apply(trimmed_var,limits=(0.1,0.1)).sort_values().tail(10)
top_ten_trim_var


# 6.5.10

import plotly.express as px

# Create horizontal bar chart of top_ten_trim_var
fig = px.bar(
    x=top_ten_trim_var,
    y=top_ten_trim_var.index,
    orientation='h',  # horizontal bars
    title="Top 10 Features by Variance"
)

# Update axis labels
fig.update_layout(
    xaxis_title="Variance",
    yaxis_title="Feature"
)

# Don't delete the code below
fig.write_image("images/6-5-10.png", scale=1, height=500, width=700)
fig.show()



# 6.5.11

high_var_cols = top_ten_trim_var.tail(5).index.to_list()
high_var_cols


# 6.5.12

X = df_small_biz[high_var_cols]
print("X shape:", X.shape)
X.head()


# 6.5.13

n_clusters = range(2, 13)
inertia_errors = []
silhouette_scores = []

# Use for loop
for k in n_clusters:
    # Build model
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    
    # Train model
    model.fit(X)
    
    # Calculate inertia
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    
    # Calculate silhouette score
    silhouette_scores.append(silhouette_score(X, model.named_steps["kmeans"].labels_))

print("Inertia:", inertia_errors[:10])
print()
print("Silhouette Scores:", silhouette_scores[:3])


# 6.5.14

import plotly.express as px

# Create the plot
fig = px.line(
    x=n_clusters, y=inertia_errors, title="Your Title"
)

# Update the layout with axis titles
fig.update_layout(
    xaxis_title="Your x_label", 
    yaxis_title="Your y_label"
)

# Save the plot as a PNG image
fig.write_image("images/6-5-14.png", scale=1, height=500, width=700)

# Show the plot
fig.show()


# 6.5.15

import plotly.express as px

# Create the plot
fig = px.line(
    x=n_clusters, y=silhouette_scores, title="Your Title"
)

# Update the layout with axis titles
fig.update_layout(
    xaxis_title="Your x_label", 
    yaxis_title="Your y_label"
)

# Save the plot as a PNG image
fig.write_image("images/6-5-15.png", scale=1, height=500, width=700)

# Show the plot
fig.show()



# 6.5.16

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create the final model with a pipeline
final_model = make_pipeline(
    StandardScaler(),             # Standardize the features
    KMeans(n_clusters=3, random_state=42)  # Apply KMeans clustering with 3 clusters
)

# Train the final model on the data X
final_model.fit(X)





# 6.5.17

labels = final_model.named_steps ["kmeans"].labels_
xgb = X.groupby(labels).mean()
xgb


# 6.5.18

# Create side-by-side bar chart of `xgb`
fig = px.bar(
xgb,
barmode="group",
title="Your Title"
)
fig.update_layout(xaxis_title="Your x_label", yaxis_title="Your y_label")
# Don't delete the code below
fig.write_image("images/6-5-18.png", scale=1, height=500, width=700)
fig.show()


# 6.5.19

pca = PCA(n_components=2, random_state=42)
X_t=pca.fit_transform(X)
X_pca = pd.DataFrame (X_t, columns=["PC1", "PC2"])

# 6.5.20

fig = px.scatter(
data_frame=X_pca,
x="PC1",
y="PC2",
color=labels.astype(str),
title="PCA Representation of Clusters"
)
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
# Don't delete the code below
fig.write_image("images/6-5-20.png", scale=1, height=500, width=700)
fig.show()



