import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv('drug_listings.csv')

# Remove missing values
df_wmv = df.dropna()

# Convert all columns types to string
df_wmv = df_wmv.astype(str)

non_numerical_chars = df_wmv['price'].str.replace(r'[0-9.]', '').unique()
# Remove these characters from the price column and add them to a new column: "currency_name"
df_wmv['currency_name'] = df_wmv['price'].str.replace(r'[0-9.]', '')

for currency in non_numerical_chars:
    print(df_wmv[df_wmv['currency_name'] == currency]['price'])


