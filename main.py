import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# suppressing FutureWarning
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load the dataset
df = pd.read_csv('drug_listings.csv')

# Remove missing values
df_wmv = df.dropna()

# Convert all columns types to string
df_wmv = df_wmv.astype(str)

non_numerical_chars = df_wmv['price'].str.replace(r'[0-9.]', '').unique()

df_wmv['currency_name'] = df_wmv['price'].str.replace(r'[0-9.]', '')

# we drop the row that contains non_numerical_chars[-1]
df_wmv = df_wmv[df_wmv['currency_name'] != non_numerical_chars[-1]]

# we drop the rows that contain items of non_numerical_chars that contains the euro sign
# and also the pound sign, the string "A$" and the string "C$"
# which are non_numerical_chars[1], non_numerical_chars[2] and non_numerical_chars[4] and non_numerical_chars[5]

df_wmv = df_wmv[~df_wmv['currency_name'].isin([non_numerical_chars[1], non_numerical_chars[2], non_numerical_chars[4], non_numerical_chars[5]])]

# we create a subset for prices in BTC and one for prices in USD
df_usd = df_wmv[df_wmv['currency_name'].isin([non_numerical_chars[0], non_numerical_chars[3], non_numerical_chars[7], non_numerical_chars[10], non_numerical_chars[11]])]
print(df_usd['price'].str.replace(r'[0-9.]', '').value_counts())
print()

# print only one instance of a price for every currency_name in USD
print(df_usd.groupby('currency_name')['price'].first())
print()

# for prices that have currency_name as non_numerical_chars[3] or non_numerical_chars[11],
# we split the value of the price column in a list by the spaces, and get the penultimate element as the price converted to float
df_usd1 = df_usd[df_usd['currency_name'].isin([non_numerical_chars[3], non_numerical_chars[7]])]
df_usd1['price'] = df_usd1['price'].str.split().apply(lambda x: float(x[-2]) if len(x) > 1 else x[0])

# for prices that have currency_name as non_numerical_chars[10] or non_numerical_chars[11],
# we split the value of the price column in a list by the spaces, and get the first element as the price converted to float
df_usd2 = df_usd[df_usd['currency_name'].isin([non_numerical_chars[10], non_numerical_chars[11]])]
df_usd2['price'] = df_usd2['price'].str.split().apply(lambda x: float(x[0]))

# for prices that have currency_name as non_numerical_chars[0]
# we remove the non-numerical characters and convert the price to float
df_usd3 = df_usd[df_usd['currency_name'] == non_numerical_chars[0]]
df_usd3['price'] = df_usd3['price'].str.replace(r'[^\d.]', '').astype(float)

# merge all three subsets and assign currency name as USD
df_usd = pd.concat([df_usd1, df_usd2, df_usd3])
df_usd['currency_name'] = 'USD'