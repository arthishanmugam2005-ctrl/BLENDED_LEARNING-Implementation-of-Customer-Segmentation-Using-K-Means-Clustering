# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import required libraries and load the Mall Customers dataset using pandas.
2.Select features (Age, Annual Income, Spending Score) and standardize the data using StandardScaler.
3.Apply K-Means clustering with different numbers of clusters (1–10) to compute inertia values.
4.Plot the Elbow Method graph to determine the optimal number of customer segments.
```
## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: Arthi S
RegisterNumber:  212225220011
*/
```
```
import os 
os.environ["OMPI NUM_THREADS"] = "1" 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn. cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
 
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

data = pd.read_csv('CustomerData.csv')

print(data.head()) 
print(data.columns)

features = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
X = data [features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10) # Explicit n init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-') 
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
```

## Output:

<img width="786" height="444" alt="9" src="https://github.com/user-attachments/assets/a3182311-4265-4958-a313-17fe4eb27188" />

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
