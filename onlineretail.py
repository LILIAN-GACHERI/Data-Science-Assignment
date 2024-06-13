#the required libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv('OnlineRetail.csv', encoding = 'latin1')
print(data.head(10))

print(data.shape)

print(data.info())

print(data.columns)

print(data.describe())

#null values
df_null =round(100*(data.isnull().sum())/len(data), 2)
data = data.dropna()
print(data.shape)
print(data.isnull().sum())

'''data.drop(["StockCode], axis= 1,inplace =True)
print(data.columns)'''

#changing customer id to to a business understanding form
data['CustomerID'] = data['CustomerID'].astype(str)
print(data.info())

#preparing the data 
#new atribute: Monetary
#creating a column named Amount by multiplying quantity and unit price
data['Amount'] = data['Quantity']*data['UnitPrice']

#customer with highest amount
data_m = data.groupby('CustomerID')['Amount'].sum().sort_values (ascending=False)

#product selling More
data_m = data.groupby('Description')['Quantity'].sum().sort_values (ascending=False)
print(data_m.head(5))

#which region is buying alot
data_m = data.groupby('Country')['Quantity'].sum().sort_values (ascending=False)
data_m = data_m.reset_index()


data_m = data.groupby('Description')['InvoiceNo'].count().sort_values (ascending=False)
print(data_m.head(5))

# Convert the 'InvoiceDate' column to datetime format
data['InvoiceDate']= pd.to_datetime(data['InvoiceDate'], format = '%m/%d/%Y %H:%M')

#compute the maximum and minimum date to know the first and last transcation date
max_date = max(data['InvoiceDate'])
print("Latest date in the dataset:", max_date)

min_date = min(data['InvoiceDate'])
print("Earliest date in the dataset:",min_date)

#the total number of days
total_days =max(data['InvoiceDate']) - min(data['InvoiceDate'])
print("Total number of days in the dataset:", total_days)

# Calculate the date, 30 days before the maximum date
date =max_date - pd.Timedelta(days=30)
print("start date of last 30 days:", date)
days = max_date-date
print(days)

# Filter the dataset for the last 30 days
last_30_days_data = data[data['InvoiceDate'] >= date]

# Calculate the total sales for the last 30 days
total_sales_last_30_days = last_30_days_data['Amount'].sum()

print("Total sales of the last 30 days:", total_sales_last_30_days)

# Calculate the total sales for all days
total_sales_all_days = data['Amount'].sum()

print("Total sales for all the days in the dataset:", total_sales_all_days)


#Buiding the model using kmeans to find thwe optimal value of k
# Assuming we are clustering based on the 'Amount' feature
features = data[['Amount']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow Method
ssd = []  # Sum of squared distances
K = range(2, 8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_features)
    ssd.append(kmeans.inertia_)

# Elbow Method graph
plt.plot(K, ssd, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()  

#the optimal value of k using elbow technique is 3.


#silhouette score

'''score = []
models = {}
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    score.append(score)
    models[k] = kmeans
print("Scores:", score)

# Plotting the Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K, score, 'bo-')  # 'bo-' means blue color, circle markers, solid line
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal values of K')
plt.show()
the optimal clusters of k is 2'''







