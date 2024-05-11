#!/usr/bin/env python
# coding: utf-8

# # Group Project – Team *Aho*
# ## TOR Drug Listings Analysis
# 
# ### Cybercrime and Fraud Detection a.y. 2023/2024
# #### LUISS Guido Carli
# 
# ### Group members
# - Tommaso Agudio –
# - Eyad Ahmed – 
# - Davide Beltrame – 268701
# - Cédric Roger – 
# - Tom Rummens – 

# ## 1. Exploratory Data Analysis

# ### 1.1. Importing librairies and tools

# In[78]:


import pandas as pd # we will use this library to work with dataframes
import numpy as np # this one to work with arrays
import matplotlib.pyplot as plt # this one to plot data
import seaborn as sns # and this one to plot data in a fancier way
import os # this library provides a way to work with the operating system
import networkx as nx # this library will be used to work with networks


# ### 1.2. Overview of the dataset

# In[79]:


# Load the dataset
df = pd.read_csv('drug_listings.csv')

# Display the first 5 rows of the dataset
df.head()


# In[80]:


# Display the number of rows and columns in the dataset
print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# Display the columns and data types of the columns
counter = 1
for i in df.columns:
    # print(f"Column {i} is {df.column name[i]} and its data type is {df.dtypes[i]}.") but fix the syntax error
    print(f"Column {counter} is {i} and its data type is {df[i].dtype}.")
    counter += 1


# In[81]:


# Display the summary statistics of the dataset
df.describe()

# the 


# In[82]:


# bar plot that shows, for each column, the number of total nonmissing values, the number of missing values, and the number of unique values
# add the numbers to the bars
total_values = df.count()
missing_values = df.isnull().sum()
unique_values = df.nunique()
df_info = pd.DataFrame({'total_values': total_values, 'missing_values': missing_values, 'unique_values': unique_values})
df_info.plot(kind='bar', figsize=(15, 5))
plt.title('Information about the columns')
plt.show()


# ### 1.3. Data Preprocessing and Cleaning

# #### 1.3.1. Handling missing values and duplicates

# In[83]:


# Display the number of missing values in each column
# print(df.isnull().sum())

# Display the percentage of missing values in each column
# and display the percentages with 2 digits after the decimal point and the % sign
print((df.isnull().sum() / df.shape[0] * 100).round(2).astype(str) + '%')

# Remove missing values
df_wmv = df.dropna()


# In[84]:


# Display the number of unique values in each column
# print(df_wmv.nunique())

# Display the percentage of unique values in each column
# and display the percentages with 2 digits after the decimal point and the % sign
print((df_wmv.nunique() / df_wmv.shape[0] * 100).round(2).astype(str) + '%')


# In[85]:


# Check for duplicate rows defined as rows with the same values in all columns
print(df_wmv.duplicated().sum())

# Create a new df without product_description column, as there could be products with the same values in all columns except product_description
df_wmv_nd = df_wmv.drop(columns=['product_description'])

# Check for duplicate rows defined as rows with the same values in all columns
print(df_wmv_nd.duplicated().sum())

# print their percentage over the total number of rows
print((df_wmv_nd.duplicated().sum() / df_wmv_nd.shape[0] * 100).round(2).astype(str) + '%')

# Print 2 rows which have the same values in all columns
df_wmv_nd[df_wmv_nd.duplicated(keep=False)].head(2)


# Duplicate rows as defined above are negligible.

# #### 1.3.2. Fixing Prices

# In[86]:


# suppressing FutureWarning
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Convert all columns types to string
df_wmv = df.dropna()
df_wmv = df_wmv.drop_duplicates()
df_wmv = df_wmv.astype(str)


# In[87]:


# We get the unique non-numerical characters in the price column
# Ensuring regex is explicitly enabled
non_numerical_chars = df_wmv['price'].str.replace(r'[0-9.]', '', regex=True).unique()
print(f'The non-numerical characters in the price column are: {non_numerical_chars}')
print()

# Print them in a list with their indexes
for i, char in enumerate(non_numerical_chars):
    print(f"Index {i}: {char}")


# In[88]:


# Remove these characters from the price column and add them to a new column: "currency_name"
df_wmv['currency_name'] = df_wmv['price'].str.replace(r'[0-9.]', '', regex=True)

# we print one instance of price for each currency_name
print(df_wmv.groupby('currency_name')['price'].first())


# In[89]:


# Display them with their frequencies
print(df_wmv['price'].str.replace(r'[0-9.]', '', regex=True).value_counts())


# In[90]:


# display their percentage over the total number of rows
print((df_wmv['price'].str.replace(r'[0-9.]', '', regex=True).value_counts() / df_wmv.shape[0] * 100).round(2).astype(str) + '%')


# In[91]:


# we drop the row that contains non_numerical_chars[-1]
df_wmv = df_wmv[df_wmv['currency_name'] != non_numerical_chars[-1]]

# we drop the rows that contain items of non_numerical_chars that contains the euro sign
# and also the pound sign, the string "A$" and the string "C$"
# which are non_numerical_chars[1], non_numerical_chars[2] and non_numerical_chars[4] and non_numerical_chars[5]

df_wmv = df_wmv[~df_wmv['currency_name'].isin([non_numerical_chars[1], non_numerical_chars[2], non_numerical_chars[4], non_numerical_chars[5]])]

print(df_wmv['price'].str.replace(r'[0-9.]', '',regex = True).value_counts())


# ##### Creating subset of USD prices

# In[92]:


# we create a subset for prices in BTC and one for prices in USD
df_usd = df_wmv[df_wmv['currency_name'].isin([non_numerical_chars[0], non_numerical_chars[3], non_numerical_chars[7], non_numerical_chars[10], non_numerical_chars[11]])]
print(df_usd['price'].str.replace(r'[0-9.]', '', regex=True).value_counts())
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
df_usd3['price'] = df_usd3['price'].str.replace(r'[^\d.]', '',regex = True).astype(float)

# merge all three subsets and assign currency name as USD
df_usd = pd.concat([df_usd1, df_usd2, df_usd3])
df_usd['currency_name'] = 'USD'


# In[93]:


# check if all prices in df_usd are numerical
print(df_usd['price'].apply(lambda x: isinstance(x, (int, float))).all())

# check if all prices in df_usd are positive
print((df_usd['price'] > 0).all())

# count and drop rows with negative prices or equal to zero
print(df_usd['price'].le(0).sum())
df_usd = df_usd[df_usd['price'] > 0]


# ##### Creating subset of BTC prices

# In[94]:


# we create a subset for prices in BTC
df_btc = df_wmv[df_wmv['currency_name'].isin([non_numerical_chars[6], non_numerical_chars[8], non_numerical_chars[9], non_numerical_chars[12]])]
print(df_btc['price'].str.replace(r'[0-9.]', '', regex = True).value_counts())
print()

# for prices that have currency_name as non_numerical_chars[6] or non_numerical_chars[8],
# we split the value of the price column in a list by the spaces, and get the penultimate element as the price converted to float
df_btc1 = df_btc[df_btc['currency_name'].isin([non_numerical_chars[6], non_numerical_chars[8]])]
df_btc1['price'] = df_btc1['price'].str.split().apply(lambda x: float(x[-2]) if len(x) > 1 else x[0])

# for prices that have currency_name as non_numerical_chars[9],
# we split the value of the price column in a list by the spaces, and get the second element as the price converted to float
df_btc2 = df_btc[df_btc['currency_name'] == non_numerical_chars[9]]
df_btc2['price'] = df_btc2['price'].str.split().apply(lambda x: float(x[1]))

# for prices that have currency_name as non_numerical_chars[12],
# we take as price the price value minus the first character
df_btc3 = df_btc[df_btc['currency_name'] == non_numerical_chars[12]]
df_btc3['price'] = df_btc3['price'].str[1:].astype(float)

# merge all three subsets and assign currency name as BTC
df_btc = pd.concat([df_btc1, df_btc2, df_btc3])
df_btc['currency_name'] = 'BTC'


# In[95]:


# check if all prices in df_btc are numerical
print(df_btc['price'].apply(lambda x: isinstance(x, (int, float))).all())

# check if all prices in df_btc are positive
print((df_btc['price'] > 0).all())

# count and drop rows with negative prices or equal to zero
print(df_btc['price'].le(0).sum())
df_btc = df_btc[df_btc['price'] > 0]


# ### 2. Insights on Prices

# #### 2.1. USD prices

# In[96]:


df_usd.head()


# In[97]:


# print the main statistics of the price column
print(df_usd['price'].describe())

# and the median
print(df_usd['price'].median())


# The mean is much higher than the median and the std is very high, which suggests the presence of outliers.

# In[98]:


# We check for outliers in the price column
sns.boxplot(x=df_usd['price'])
plt.title('Boxplot of prices in USD')


# In[99]:


# we remove outlier as values greater than 3xiqr
q1 = df_usd['price'].quantile(0.25)
q3 = df_usd['price'].quantile(0.75)
iqr = q3 - q1
df_usd = df_usd[(df_usd['price'] >= q1 - 3 * iqr) & (df_usd['price'] <= q3 + 3 * iqr)]

# we plot the boxplot again
sns.boxplot(x=df_usd['price'])
plt.title('Boxplot of prices without outliers in USD')

# and the main statistics of the price column, including the median
print(df_usd['price'].describe())
print(df_usd['price'].median())


# In[100]:


# we plot the distribution of the price column
sns.histplot(df_usd['price'], kde=True)
plt.title('Distribution of prices in USD')


# #### 2.2. BTC prices

# In[101]:


df_btc.head()


# In[102]:


# print the main statistics of the price column
print(df_btc['price'].describe())

# and the median
print(df_btc['price'].median())


# The mean is much lower than the median, which suggests the presence of outliers.

# In[103]:


# We check for outliers in the price column
sns.boxplot(x=df_btc['price'])
plt.title('Boxplot of prices in BTC')

# and we print the 50 highest prices, just the values of the price column
print(df_btc['price'].nlargest(100))


# In[104]:


# we remove outlier as values greater than 2
df_btc = df_btc[df_btc['price'] <= 2]

# we plot the boxplot again
sns.boxplot(x=df_btc['price'])
plt.title('Boxplot of prices without outliers in BTC')

# and the main statistics of the price column, including the median
print(df_btc['price'].describe())
print(df_btc['price'].median())


# In[105]:


# we plot the distribution of the price column
sns.histplot(df_btc['price'], kde=True)
plt.title('Distribution of prices in BTC')


# In[126]:


# we merge the two dataframes
df_final = pd.concat([df_usd, df_btc])

# we need to convert the prices that have USD as currency_name to BTC and convert their prices
# we use the conversion rate of 1 BTC = 50000 USD

# we convert the prices that have USD as currency_name to BTC
df_final.loc[df_final['currency_name'] == 'USD', 'price'] = df_final['price'] / 50000

# we convert the currency_name to BTC
df_final.loc[df_final['currency_name'] == 'USD', 'currency_name'] = 'BTC'

# and the main statistics of the price column, including the median
print(df_final['price'].describe())

# and the median
print(df_final['price'].median())

# we plot the distribution of the price column
sns.histplot(df_final['price'], kde=True)


# In[127]:


# we drop all prices under 2*10**-6 BTC, corresponding to 10 USD cents
df_final = df_final[df_final['price'] >= 10**-6]

# and the main statistics of the price column, including the median
print(df_final['price'].describe())

# and the median
print(df_final['price'].median())

# we plot the distribution of the price column
sns.histplot(df_final['price'], kde=True)
plt.title('Distribution of all prices')


# In[128]:


df_final.head()


# ## 2. Text fields extraction

# ### 2.1. Drug types

# In[130]:


# we use the following dictionary to find corresponding drugs in product_title and product_description
drug_synonyms = {
    'cannabis': ['marijuana', 'weed', 'pot', 'ganja', 'herb', 'grass', 'mary jane', 'dope', 'bud', 'reefer', 'hash', 'hashish', 'joint', 'blunt', 'spliff', 'chronic', 'dank'],
    'ecstasy': ['MDMA', 'molly', 'E', 'X', 'XTC', 'Adam', 'hug drug', 'love drug', 'beans', 'rolls', 'scooby snacks'],
    'stimulants': ['amphetamines', 'speed', 'uppers', 'adderall', 'dexies', 'bennies', 'black beauties', 'coke'],
    'psychedelics': ['LSD', 'acid', 'magic mushrooms', 'shrooms', 'psilocybin', 'DMT', 'ayahuasca', 'mescaline', 'peyote'],
    'benzes': ['Xanax', 'Valium', 'Ativan', 'Klonopin', 'roofies'],
    'prescription': ['Rx drugs', 'meds', 'pharmaceuticals', 'pills', 'tablets', 'capsules'],
    'opioids': ['heroin', 'OxyContin', 'oxycodone', 'percocet', 'vicodin', 'hydrocodone', 'codeine', 'morphine', 'fentanyl', 'tramadol'],
    'steroids': ['anabolic steroids', 'juice', 'roids', 'gear', 'pumpers'],
    'dissociative': ['ketamine', 'special K', 'K', 'vitamin K', 'DXM', 'PCP', 'angel dust'],
    'paraphernalia': ['gear', 'equipment', 'tools', 'supplies', 'kits', 'accessories'],
    'weight loss': ['diet pills', 'slimming pills', 'fat burners', 'appetite suppressants'],
    'tobacco': ['cigarettes', 'cigs', 'smokes', 'cancer sticks', 'nicotine'],
    'cocaine': ['coke', 'blow', 'snow', 'powder', 'white', 'yayo', 'charlie', 'nose candy'],
    'heroin': ['smack', 'junk', 'horse', 'H', 'dope', 'brown', 'tar', 'china white', 'skag'],
    'meth': ['methamphetamine', 'crystal meth', 'ice', 'crystal', 'glass', 'tina', 'bath salts', 'spice', 'K2']
}

# we create a new column called "drug_category" and assign the value "others" to it
df_wmv['drug_type'] = 'others'

# we iterate over the dictionary and assign the key to the rows that contain the corresponding values in product_title and product_description
for drug, synonyms in drug_synonyms.items():
    df_wmv.loc[df_wmv['product_title'].str.contains('|'.join(synonyms), case=False), 'drug_type'] = drug
    df_wmv.loc[df_wmv['product_description'].str.contains('|'.join(synonyms), case=False), 'drug_type'] = drug

# we print the number of rows for each drug category
print(df_wmv['drug_type'].value_counts())


# In[131]:


# we plot the number of rows for each drug category
df_wmv['drug_type'].value_counts().plot(kind='bar')
plt.title('Number of rows for each drug type')


# ### 2.2. Drug types prices

# In[132]:


# we do the same for the df_final dataframe

# we create a new column called "drug_category" and assign the value "others" to it
df_final['drug_type'] = 'others'

# we iterate over the dictionary and assign the key to the rows that contain the corresponding values in product_title and product_description
for drug, synonyms in drug_synonyms.items():
    df_final.loc[df_final['product_title'].str.contains('|'.join(synonyms), case=False), 'drug_type'] = drug
    df_final.loc[df_final['product_description'].str.contains('|'.join(synonyms), case=False), 'drug_type'] = drug

# we print the number of rows for each drug category
print(df_final['drug_type'].value_counts())


# In[136]:


# we check the average and median price for each drug category
# and print them in descending order
df_final.groupby('drug_type')['price'].agg(['mean', 'median']).sort_values(by='mean', ascending=False)


# In[137]:


# let's plot the averages and medians
df_final.groupby('drug_type')['price'].agg(['mean', 'median']).sort_values(by='mean', ascending=False).plot(kind='bar')
plt.title('Average and median prices for each drug type')
plt.ylabel('Price')
plt.show()


# ## 3. Sellers and cartels

# In[187]:


# we not check the top sellers by units sold
# and print the top 10 sellers
df_final['seller'].value_counts().head(15)


# In[151]:


# let's plot the distribution of units sold by seller
df_final['seller'].value_counts().plot(kind='hist', bins=100)
plt.title('Distribution of units sold by seller')


# In[168]:


# now we find the drug type with the most sellers

# we group by drug type and seller, and count the number of rows
df_sellers = df_final.groupby(['drug_type', 'seller']).size().reset_index(name='counts')

# we group by drug type and count the number of unique sellers
df_sellers = df_sellers.groupby('drug_type')['seller'].nunique().reset_index(name='unique_sellers')

# we sort the values in descending order
df_sellers.sort_values(by='unique_sellers', ascending=False)

# we plot the number of unique sellers for each drug type
df_sellers.sort_values(by='unique_sellers', ascending=False).plot(x='drug_type', y='unique_sellers', kind='bar')
plt.title('Number of unique sellers for each drug type')
plt.ylabel('Number of unique sellers')
plt.show()


# In[199]:


# the top sellers are
top_sellers = []
for i in df_final['seller'].value_counts().head(15).index:
    top_sellers.append(i)

# we compare the average prices of heroin and meth by top seller and compare them with the average prices in df_final
# we create a subset for heroin and one for meth
df_heroin = df_final[df_final['drug_type'] == 'heroin']

# we create a subset for meth
df_meth = df_final[df_final['drug_type'] == 'meth']

# we compare the average prices of heroin and meth by top seller
# and compare them with the average prices in df_final
for i in top_sellers:
    print(f'Average price for heroin by {i}: {df_heroin[df_heroin["seller"] == i]["price"].mean()}')
    print(f'Average price for meth by {i}: {df_meth[df_meth["seller"] == i]["price"].mean()}')
    print(f'Average price for heroin in df_final: {df_heroin["price"].mean()}')
    print(f'Average price for meth in df_final: {df_meth["price"].mean()}')
    print()


# In[198]:


# we plot the distribution of prices for heroin and meth
sns.histplot(df_heroin['price'], kde=True, color='blue', label='Heroin')
sns.histplot(df_meth['price'], kde=True, color='red', label='Meth')
plt.title('Distribution of prices for heroin and meth')
plt.legend()
plt.show()


# In[157]:


# to gain more insights, we print the seller with minimum, maximum, average and median units sold (times they're present in the seller column)
print(f'The fewest units sold is: {df_final["seller"].value_counts().min()}')
print(f'The most units sold is: {df_final["seller"].value_counts().max()}')
print(f'The average units sold is: {df_final["seller"].value_counts().mean()}')
print(f'The median units sold is: {df_final["seller"].value_counts().median()}')

# how many sellers sold just 1 unit
print((df_final['seller'].value_counts() == 1).sum())
# how many sellers sold less than the average units sold (12)
print((df_final['seller'].value_counts() < df_final['seller'].value_counts().mean()).sum())
# how many sellers sold more than the average units sold (12)
print((df_final['seller'].value_counts() > df_final['seller'].value_counts().mean()).sum())


# ## 4. Shipping origins and destinations

# In[74]:


# we print the number of unique values in ship_from and ship_to
print(f'The number of unique values in ship_from is {df_wmv["ship_from"].nunique()} and in ship_to is {df_wmv["ship_to"].nunique()}.')

# we print the 10 most common countries in ship_from
print(df_wmv['ship_from'].value_counts().head(10))

# we plot the 15 most common countries in ship_from
df_wmv['ship_from'].value_counts().head(15).plot(kind='bar')


# In[210]:


# we create a directed graph using the countries in ship_from and ship_to
# nodes are countries, edges are connections from ship_from to ship_to
G = nx.DiGraph()

# we add the countries in ship_from and ship_to as nodes
G.add_nodes_from(df_wmv['ship_from'].unique())
G.add_nodes_from(df_wmv['ship_to'].unique())

# we add the connections from ship_from to ship_to as edges
for i in range(df_wmv.shape[0]):
    G.add_edge(df_wmv['ship_from'].iloc[i], df_wmv['ship_to'].iloc[i])

# we plot the graph

plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue')
plt.title('Graph of connections from ship_from to ship_to')
plt.show()


# In[216]:


# we compute in degree and outdegree for each country
# in degree defined as number of times the country appears in ship_to,
# out degree defined as number of times the country appears in ship_from
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

# we plot the countries with the highest in degree
df_in_degree = pd.DataFrame(in_degree.items(), columns=['country', 'in_degree'])
df_in_degree.sort_values(by='in_degree', ascending=False).head(20).plot(x='country', y='in_degree', kind='bar')
plt.title('Countries with the highest in degree')
plt.ylabel('In degree')
plt.show()


# In[215]:


# we plot the countries with the highest out degree
df_out_degree = pd.DataFrame(out_degree.items(), columns=['country', 'out_degree'])
df_out_degree.sort_values(by='out_degree', ascending=False).head(20).plot(x='country', y='out_degree', kind='bar')
plt.title('Countries with the highest out degree')
plt.ylabel('Out degree')
plt.show()


# # NLP Analysis

# In[ ]:


# NLP Analysis of the 'product_title' column
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# running pip install wordcloud if not installed
from wordcloud import WordCloud



# Download the NLTK resources 
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


# In[ ]:


# Create a list of stopwords
stop_words = set(stopwords.words('english'))

# Create a WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Create a Porter Stemmer
stemmer = PorterStemmer()

# Tokenize the 'product_title' column
# first convert into string
df['product_title'] = df['product_title'].astype(str)
df['product_title_tokens'] = df['product_title'].apply(word_tokenize)
# TypeError: expected string or bytes-like object


# Remove stopwords from the 'product_title' column
df['product_title_tokens'] = df['product_title_tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Lemmatize the 'product_title' column
df['product_title_tokens'] = df['product_title_tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Stem the 'product_title' column
df['product_title_tokens'] = df['product_title_tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

# Create a WordCloud of the 'product_title' column
text = ' '.join(df['product_title'])
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


# Visualize the distribution of the 'product_title' column
plt.figure(figsize=(10, 5))
sns.histplot(df['product_title_tokens'].apply(len), kde=True)
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.title('Distribution of Number of Words in Product Title')
plt.show()


# In[ ]:


# print more statistics on the number and type of words in product titles
# Create a new column with the number of words in the 'product_title' column
df['num_words'] = df['product_title_tokens'].apply(len)

# Display the average number of words in the 'product_title' column
print('Average Number of Words in Product Title:', df['num_words'].mean())

# Display the maximum number of words in the 'product_title' column
print('Maximum Number of Words in Product Title:', df['num_words'].max())

# Display the minimum number of words in the 'product_title' column
print('Minimum Number of Words in Product Title:', df['num_words'].min())

# Display the most common words in the 'product_title' column
# exclude special characters and stopwords

# Create a list of special characters
special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/']

# Create a list of words without special characters
df['product_title_tokens'] = df['product_title_tokens'].apply(lambda x: [word for word in x if word not in special_chars])

words = [word for sublist in df['product_title_tokens'] for word in sublist]
word_freq = nltk.FreqDist(words)
print('Most Common Words in Product Title:', word_freq.most_common(50))


# In[ ]:


# Classification of fraudolent listings using NLP insights
# we classify listings as fraudulent when there are exclamation marks and misspelled words in the product title, such as qualiti

# Create a new column with the number of exclamation marks in the 'product_title' column
df['num_exclamation_marks'] = df['product_title'].apply(lambda x: x.count('!'))

# Create a new column with the number of misspelled words in the 'product_title' column
df['num_misspelled_words'] = df['product_title_tokens'].apply(lambda x: sum([1 for word in x if 'qualiti' in word]))

# Create a new column to classify listings as fraudulent or not
df['is_fraudulent'] = (df['num_exclamation_marks'] > 0) | (df['num_misspelled_words'] > 0)

# Display the count of fraudulent and non-fraudulent listings
print(df['is_fraudulent'].value_counts())

# Display the percentage of fraudulent and non-fraudulent listings
print(df['is_fraudulent'].value_counts(normalize=True))

# True means that the listing is fraudulent, while False means that the listing is not fraudulent.

