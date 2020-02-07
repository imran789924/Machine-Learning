#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:49:21 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4:5].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X[:, 0] = encoder.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()

X = np.delete(X, 1, 1)



from sklearn.model_selection import train_test_split
X_train, X_teast, y_train, y_test = train_test_split(X, y, test_size=.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_teast)


import statsmodels.api as sm
X = np.append(arr=np.ones((200,1), dtype=int ), values = X, axis = 1)

X_opt = X[:, [3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


correlation = dataset.corr()


'''
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('shopping data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.draw()
'''