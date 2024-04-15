import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv('drug_listings.csv')

# Display the first 5 rows of the dataset
print(df.head())

# Display the last 5 rows of the dataset
print(df.tail())

# Display the shape of the dataset
print(df.shape)

# Display the columns of the dataset
print(df.columns)

# Display the data types of the columns
print(df.dtypes)

# Display the number of missing values in each column
print(df.isnull().sum())

# Display the number of unique values in each column
print(df.nunique())

# Display the summary statistics of the dataset
print(df.describe())

# Display the correlation matrix of the dataset
print(df.corr())

# Display the count of each unique value in the 'product_title' column
print(df['product_title'].value_counts())
