import gzip
import json

import pandas as pd
import wqet_grader
from IPython.display import VimeoVideo

wqet_grader.init("Project 5 Assessment")


# 5.1.2

%%bash
gunzip -c data/taiwan-bankruptcy-data-test-features.json.gz | head -n 10

# 5.1.3

# 5.1.4

# 5.1.5

import json

# Load the JSON data
with open("data/poland-bankruptcy-data-2009.json", "r") as f:
    data = json.load(f)

# Print structure of each key
for key in data:
    print(f"\nKey: {key}")
    print("Type:", type(data[key]))
    if isinstance(data[key], list):
        print("First 2 items:")
        for item in data[key][:2]:
            print(item)
    elif isinstance(data[key], dict):
        print("First 2 keys:", list(data[key].keys())[:2])
    else:
        print("Value:", data[key])

df = pd.DataFrame(data["data"])
df.head()


# 5.1.6

import json

# Using context manager to open the JSON file and load it as a dictionary
with open("data/poland-bankruptcy-data-2009.json", "r") as file:
    poland_data = json.load(file)

# Display the loaded data (optional)
print(poland_data)


# 5.1.7

# Print `poland_data` keys
# Print the keys of the poland_data dictionary
print(poland_data.keys())



# 5.1.8

# Continue Exploring `poland_data`
# Print the keys and their corresponding values in the poland_data dictionary
for key in poland_data:
    print(f"\nKey: {key}")
    print("Type:", type(poland_data[key]))
    if isinstance(poland_data[key], list):
        print("First 2 items:", poland_data[key][:2])  # Print the first 2 items if it's a list
    elif isinstance(poland_data[key], dict):
        print("First 2 keys:", list(poland_data[key].keys())[:2])  # Print the first 2 keys if it's a dict
    else:
        print("Value:", poland_data[key])

        
# If the 'data' key exists and is a list, check the first few records
if 'data' in poland_data and isinstance(poland_data['data'], list):
    print("\nFirst 2 records in 'data':")
    for record in poland_data['data'][:2]:  # Print the first 2 records
        print(record)


# 5.1.9

# Calculate number of companies
# Calculate the number of companies
num_companies = len(poland_data['data'])
print(f"Number of companies in the dataset: {num_companies}")


# 5.1.10

# Calculate number of features
# Get the first company (company_1) record from the 'data' key
company_1 = poland_data['data'][0]

# Calculate the number of features associated with company_1
num_features = len(company_1)
print(f"Number of features for 'company_1': {num_features}")


# 5.1.11

# Iterate through companies
# Get the number of features for the first company
num_features = len(poland_data['data'][0])

# Check that all companies have the same number of features
for idx, company in enumerate(poland_data['data']):
    if len(company) != num_features:
        print(f"Company {idx+1} does not have the same number of features!")
    else:
        print(f"Company {idx+1} has {num_features} features.")

# 5.1.12

import gzip
import json

# Using a context manager to open the gzipped JSON file and load it into a dictionary
with gzip.open("data/poland-bankruptcy-data-2009.json.gz", "rt", encoding="utf-8") as f:
    poland_data_gz = json.load(f)

# Print the keys to verify the data
print(poland_data_gz.keys())


# 5.1.13

# Explore `poland_data_gz`
# Compare the keys of both dictionaries
print("Keys in poland_data:", poland_data.keys())
print("Keys in poland_data_gz:", poland_data_gz.keys())

# Check the first few records under 'data' key for both dictionaries
print("First record in poland_data['data']:", poland_data['data'][0])
print("First record in poland_data_gz['data']:", poland_data_gz['data'][0])

# Confirm the number of entries in both 'data' keys
print(f"Number of companies in poland_data: {len(poland_data['data'])}")
print(f"Number of companies in poland_data_gz: {len(poland_data_gz['data'])}")
