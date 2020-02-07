# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:16:04 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('F:\Machine Learning\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()
X=scale_X.fit_transform(X)
y=scale_y.fit_transform(y)


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)


plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Salary Table According to level of employee')
plt.xlabel('employee level')
plt.ylabel('salary')
plt.show()

#smother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Smoother table for salary vs designation level')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()


y_pred= scale_y.inverse_transform(regressor.predict(scale_X.transform(np.array([[6.5]]))))