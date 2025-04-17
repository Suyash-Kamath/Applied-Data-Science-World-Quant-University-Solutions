import wqet_grader

wqet_grader.init("Project 5 Assessment")
# Import Library


# Import Libraries here
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
import gzip
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wqet_grader
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
ConfusionMatrixDisplay,
classification_report,
confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import ipywidgets as widgets
from ipywidgets import interact
from sklearn.ensemble import GradientBoostingClassifier
from teaching_tools.widgets import ConfusionMatrixWidget


# Prepare Data
# 5.5.1

with gzip.open("data/taiwan-bankruptcy-data.json.gz","r") as read_file:
    taiwan_data = json.load(read_file)

print(type(taiwan_data))

# 5.5.2

taiwan_data_keys = taiwan_data.keys()
print(taiwan_data_keys)

# 5.5.3

n_companies = len(taiwan_data["observations"])
print(n_companies)

# 5.5.4

n_features = len(taiwan_data["observations"][0])
print(n_features)

# 5.5.5

# # Create wrangle function
def wrangle(filename):
    with gzip.open(filename, "r") as f:
        data = json.load(f)
    return pd.DataFrame().from_dict(data["observations"]).set_index("id")

df = wrangle("data/taiwan-bankruptcy-data.json.gz")
print("df shape:", df.shape)
df.head()


# 5.5.6

df.info()


# In[24]:


nans_by_col =  df.isnull().sum()
nans_by_col = pd.Series(nans_by_col)
print("nans_by_col shape:", nans_by_col.shape)
nans_by_col.head()


# 5.5.7

df["bankrupt"].value_counts(normalize=True).plot(
    kind="bar",
    xlabel="Bankrupt",
    ylabel="Frequency",
    title="Class Balance"
);
# Don't delete the code below 👇
plt.savefig("images/5-5-7.png", dpi=150)


# 5.5.8

target = "bankrupt"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)


# 5.5.9


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_train.shape)
print("y_test shape:", y_train.shape)


# 5.5.10

over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()

# 5.5.11

clf = GradientBoostingClassifier(random_state=42)
clf


# 5.5.12

cv_scores = cross_val_score(clf, X_train_over, y_train_over, cv=5, n_jobs=-1)
print(cv_scores)

# Ungraded Task

params=params={
    "n_estimators":range(20,31,5),
    "max_depth":range(2,5)
}
params

# 5.5.13

model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
model

# Ungraded tASK

model.fit(X_train_over, y_train_over)

# 5.5.14
cv_results = pd.DataFrame(model.cv_results_)
cv_results.head(5)


# 5.5.15

best_params = model.best_params_
print(best_params)

# Ungrade Task

acc_train = model.score(X_train_over, y_train_over)
acc_test = model.score(X_test, y_test)

print("Model Training Accuracy:", round(acc_train, 4))
print("Model Test Accuracy:", round(acc_test, 4))

# 5.5.16

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);
# Don't delete the code below 👇
plt.savefig("images/5-5-16.png", dpi=150)


# 5.5.17



from sklearn.metrics import classification_report
class_report = classification_report(y_test, model.predict(X_test))
print(class_report)

# 5.5.18

features = X_train_over.columns
importances = model.best_estimator_.named_steps["randomforestclassifier"].feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()
feat_imp.tail(10).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");
# Don't delete the code below 👇
plt.savefig("images/5-5-17.png", dpi=150)



# 5.5.19

# Save model
with open("model-5-5.pkl", "wb") as f:
    pickle.dump(model, f)


# 5.5.20

!cat /home/jovyan/work/ds-curriculum/050-bankruptcy-in-poland/my_predictor_assignment.py
# next cell me daalo

# Import your module
from my_predictor_assignment import make_predictions

# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/taiwan-bankruptcy-data-test-features.json.gz",
    model_filepath="model-5-5.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()


# 
