#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:51:39 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, [3,4]].values


import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendogram for selecting number of clusters')
plt.xlabel('customers')
plt.ylabel('Euclidean distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_clusters = hc.fit_predict(X)


#cluster visualization
plt.scatter(X[y_clusters==0, 0], X[y_clusters==0 ,1], c='red', s=100, label='careful')
plt.scatter(X[y_clusters==1, 0], X[y_clusters==1 ,1], c='blue', s=100, label='standard')
plt.scatter(X[y_clusters==2, 0], X[y_clusters==2 ,1], c='green', s=100, label='target')
plt.scatter(X[y_clusters==3, 0], X[y_clusters==3 ,1], c='cyan', s=100, label='careless')
plt.scatter(X[y_clusters==4, 0], X[y_clusters==4 ,1], c='magenta', s=100, label='Sensible')
#plt.scatter(kMeans.cluster_centers_[:,0], kMeans.cluster_centers_[:, 1], c='yellow', s=300, label='cluster Centers')
plt.title('spending')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.draw()
