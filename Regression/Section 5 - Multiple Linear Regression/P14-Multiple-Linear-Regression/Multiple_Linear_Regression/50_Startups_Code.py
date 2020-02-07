# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:01:08 2019

@author: imran
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1 ].values
y = dataset.iloc[ : , 4 ].values


'''
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder_X = LabelEncoder()
X.iloc[:, 3] = labelEncoder_X.fit_transform(X.iloc[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()
'''
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
y_pred = linearRegression.predict(X_test)



import statsmodels.api as sm
X = np.append(arr = np.ones((50,1), dtype=int), values = X, axis = 1)
#X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#OLS = ordinary least square
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=.2, random_state=0)

linearRegression1 = LinearRegression()
linearRegression1.fit(X_train, y_train)
y_pred2 = linearRegression1.predict(X_test)