# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:38:47 2019

@author: imran
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1 ].values
y = dataset.iloc[ : , 4 ].values


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()


#avoid dummy variable trap, this is optional as libraries handle this automatically
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


pred_y = regressor.predict(X_test)