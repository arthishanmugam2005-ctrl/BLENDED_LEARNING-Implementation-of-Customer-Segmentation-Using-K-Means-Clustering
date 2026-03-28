# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import required libraries such as pandas, seaborn, matplotlib, and sklearn.
2.Load the dataset CustomerData.csv and Select important features (Age, Annual Income, Spending Score).
3.Apply StandardScaler to normalize the feature values.
4.Use the Elbow Method to determine the optimal number of clusters.
5.Train the K-Means clustering model with the optimal number of clusters.
6.Assign cluster labels to each data point.
7.Calculate the Silhouette Score to evaluate clustering performance.
8.Visualize the clusters using a scatter plot.
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
data=pd.read_csv('CustomerData (1).csv')
print(data.head())
print(data.columns)
features=['Age','Annual Income (k$)','Spending Score (1-100)']
X=data[features]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,4))
plt.plot(range(1,11),wcss,marker='o',linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
optimal_clusters=4
kmeans=KMeans(n_clusters=optimal_clusters,random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print("Name:ARTHI S")
print("Register no:212225220011")
print(f'Silhouette Score: {sil_score}')

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7
)

plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show() 
```

## Output:
<img width="609" height="163" alt="Screenshot 2026-03-28 192546" src="https://github.com/user-attachments/assets/36a75b56-3ca8-48db-9370-1438f2b88021" />
<img width="636" height="330" alt="Screenshot 2026-03-28 192609" src="https://github.com/user-attachments/assets/3b5cb4d4-e9b7-4701-b107-eed5e13728cf" />
<img width="635" height="468" alt="Screenshot 2026-03-28 192627" src="https://github.com/user-attachments/assets/d3391495-3f74-4393-831c-627c18a7c4ba" />

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
