import warnings

import wqet_grader

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 4 Assessment")

import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from category_encoders import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Prepare data : Connect

%load_ext sql
%sql sqlite:////home/jovyan/nepal.sqlite

# 4.5.1

%%sql
SELECT distinct(district_id)
FROM id_map

result = _.DataFrame().squeeze()  # noqa F821

wqet_grader.grade("Project 4 Assessment", "Task 4.5.1", result)

%%sql
SELECT *
FROM building_damage
LIMIT 5
      
# 4.5.2
%%sql
SELECT count(*)
FROM id_map
WHERE district_id = 1

result = [_.DataFrame().astype(float).squeeze()]  # noqa F821
wqet_grader.grade("Project 4 Assessment", "Task 4.5.2", result)

      
# 4.5.3

%%sql
SELECT count(*)
FROM id_map
WHERE district_id = 3

result = [_.DataFrame().astype(float).squeeze()]  # noqa F821
wqet_grader.grade("Project 4 Assessment", "Task 4.5.3", result)


# 4.5.4

%%sql
SELECT distinct(i.building_id) AS b_id,
s.*,
d.damage_grade
FROM id_map AS i
JOIN building_structure AS s ON i.building_id = s.building_id
JOIN building_damage AS d ON i.building_id = d.building_id
WHERE district_id = 3
LIMIT 5


# 4.5.5

import sqlite3
import pandas as pd

def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
        SELECT DISTINCT(i.building_id) AS b_id,
               s.*,
               d.damage_grade
        FROM id_map AS i
        JOIN building_structure AS s ON i.building_id = s.building_id
        JOIN building_damage AS d ON i.building_id = d.building_id
        WHERE district_id = 3
    """

    # Read query results into DataFrame
    df = pd.read_sql(query, conn, index_col="b_id")

    # Identify leaky columns
    drop_cols = [col for col in df.columns if "post_eq" in col]

    # Create binary target
    df["damage_grade"] = df["damage_grade"].str[-1].astype(int)
    df["severe_damage"] = (df["damage_grade"] > 3).astype(int)

    # Drop old target
    drop_cols.append("damage_grade")

    # Drop multicollinearity column
    drop_cols.append("count_floors_pre_eq")

    # Drop high cardinality categorical column
    drop_cols.append("building_id")

    return df.drop(columns=drop_cols)

df = wrangle("/home/jovyan/nepal.sqlite")
df.head()


wqet_grader.grade(
    "Project 4 Assessment", "Task 4.5.5", wrangle("/home/jovyan/nepal.sqlite")
)

# Create correlation matrix
correlation = df.select_dtypes("number").drop(columns="severe_damage").corr()
correlation
# Plot heatmap of `correlation`
sns.heatmap(correlation)



# 4.5.6
import matplotlib.pyplot as plt

# Plot class balance
df["severe_damage"].value_counts(normalize=True).plot(kind='bar')
plt.title("Kavrepalanchok, Class Balance")
plt.xlabel("Severe Damage (0 = No, 1 = Yes)")
plt.ylabel("Proportion")

# Save the plot
plt.savefig("images/4-5-6.png", dpi=150)
plt.close()  # Closes the figure to avoid overlap if plotting again


with open("images/4-5-6.png", "rb") as file:
    wqet_grader.grade("Project 4 Assessment", "Task 4.5.6", file)

# 4.5.7

# Create boxplot
sns.boxplot(x="severe_damage", y="plinth_area_sq_ft", data=df)
# Label axes
plt.xlabel("Severe Damage")
plt.ylabel("Plinth Area [sq. ft.]")
plt.title("Kavrepalanchok, Plinth Area vs Building Damage");
# Don't delete the code below
plt.savefig("images/4-5-7.png", dpi=150)

with open("images/4-5-7.png", "rb") as file:
    wqet_grader.grade("Project 4 Assessment", "Task 4.5.7", file)


# 4.5.8

roof_pivot = pd.pivot_table(
df, index="roof_type",values="severe_damage",aggfunc=np.mean
).sort_values(by="severe_damage")
roof_pivot

wqet_grader.grade("Project 4 Assessment", "Task 4.5.8", roof_pivot)

# 4.5.9

target = "severe_damage"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)

wqet_grader.grade("Project 4 Assessment", "Task 4.5.9a", X)

wqet_grader.grade("Project 4 Assessment", "Task 4.5.9b", y)


# 4.5.10

X_train, X_val, y_train, y_val = train_test_split(
X,y, test_size = 0.2, random_state = 42
)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

wqet_grader.grade("Project 4 Assessment", "Task 4.5.10", [X_train.shape == (61226, 11)])


# 4.5.11

acc_baseline = y_train.value_counts(normalize = True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))


# 4.5.12

model_lr = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=3000)
)
model_lr.fit(X_train,y_train)


# 4.5.13

lr_train_acc = model_lr.score(X_train,y_train)
lr_val_acc = model_lr.score(X_val,y_val)
print("Logistic Regression, Training Accuracy Score:", lr_train_acc)
print("Logistic Regression, Validation Accuracy Score:", lr_val_acc)

submission = [lr_train_acc, lr_val_acc]
wqet_grader.grade("Project 4 Assessment", "Task 4.5.13", submission)

# 4.5.14

depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []
for d in depth_hyperparams:
    model_dt = make_pipeline(
    OrdinalEncoder(),
      DecisionTreeClassifier(max_depth=d, random_state=42)
          )
    model_dt.fit(X_train, y_train)
    training_acc.append(model_dt.score(X_train,y_train))
    validation_acc.append(model_dt.score(X_val,y_val))
    
print("Training Accuracy Scores:", training_acc[:3])
print("Validation Accuracy Scores:", validation_acc[:3])



submission = pd.Series(validation_acc, index=depth_hyperparams)

wqet_grader.grade("Project 4 Assessment", "Task 4.5.14", submission)


# 4.5.15

 # Plot `depth_hyperparams`, `training_acc`
plt.plot(depth_hyperparams, training_acc, label="training")
plt.plot(depth_hyperparams, validation_acc, label="validation")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.title("Validation Curve, Decision Tree Model")
plt.legend();
# Don't delete the code below
plt.savefig("images/4-5-15.png", dpi=150)


# 4.5.16

 final_model_dt = make_pipeline(
OrdinalEncoder(),
DecisionTreeClassifier(max_depth=10, random_state=42)
)
final_model_dt.fit(X_val,y_val)

# 4.5.17

X_test = pd.read_csv("data/kavrepalanchok-test-features.csv", index_col="b_id")
y_test_pred = final_model_dt.predict(X_test)
y_test_pred[:5]


submission = pd.Series(y_test_pred)
wqet_grader.grade("Project 4 Assessment", "Task 4.5.17", submission)


# 4.5.18

# Get feature names
features = X_train.columns

# Get feature importances from the trained Decision Tree
importances = final_model_dt.named_steps["decisiontreeclassifier"].feature_importances_

# Create a Series with feature importances and sort them
feat_imp = pd.Series(importances, index=features).sort_values()

# Display the least important features
feat_imp.head()


# 4.5.19

# Create horizontal bar chart of feature importances

# Don't delete the code below 👇
plt.tight_layout()
plt.savefig("images/4-5-19.png", dpi=150)



  
  
