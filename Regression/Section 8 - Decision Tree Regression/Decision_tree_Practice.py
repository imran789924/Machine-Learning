# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:35:28 2020

@author: imran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('F:\Machine Learning\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict(np.array([[6.5]]))


#smother visualization

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision tree')
plt.xlabel('employee level')
plt.ylabel('salary')
plt.show()