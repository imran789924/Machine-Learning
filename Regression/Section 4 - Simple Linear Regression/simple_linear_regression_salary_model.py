# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:37:17 2019

@author: imran
"""

import numpy as np
from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.rcParams['backend'] = "Qt4Agg"
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[ : , : -1]
y = dataset.iloc[ : , 1 : ]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 1)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.scatter(X_test, y_pred, color = 'green')
plt.title('Training set\'s graph')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.draw()
