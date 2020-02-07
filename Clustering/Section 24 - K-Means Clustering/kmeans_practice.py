#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:34:45 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kMeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kMeans.fit(X)
    wcss.append(kMeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('no. of centroids')
plt.ylabel('wcss')
plt.show()


kMeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
#kMeans.fit(X)

y_kMeans = kMeans.fit_predict(X)


#cluster visualization
plt.scatter(X[y_kMeans==0, 0], X[y_kMeans==0 ,1], c='red', s=100, label='careful')
plt.scatter(X[y_kMeans==1, 0], X[y_kMeans==1 ,1], c='blue', s=100, label='standard')
plt.scatter(X[y_kMeans==2, 0], X[y_kMeans==2 ,1], c='green', s=100, label='target')
plt.scatter(X[y_kMeans==3, 0], X[y_kMeans==3 ,1], c='cyan', s=100, label='careless')
plt.scatter(X[y_kMeans==4, 0], X[y_kMeans==4 ,1], c='magenta', s=100, label='Sensible')
plt.scatter(kMeans.cluster_centers_[:,0], kMeans.cluster_centers_[:, 1], c='yellow', s=300, label='cluster Centers')
plt.title('spending')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.draw()