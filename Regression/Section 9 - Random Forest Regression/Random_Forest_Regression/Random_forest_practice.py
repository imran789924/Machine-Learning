# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:42:22 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('F:\Machine Learning\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 9 - Random Forest Regression\Random_Forest_Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)


pred_y = regressor.predict(np.array([[6.5]]))


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')