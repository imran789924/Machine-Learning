#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:18:23 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])
le_2 = LabelEncoder()
X[:, 2] = le_2.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)






