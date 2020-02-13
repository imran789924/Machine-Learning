# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
le_1 = LabelEncoder()
X[:, 2] = le_1.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]

'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=140)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
scores_array = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
scores_array.mean()
scores_array.std()


from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate' : [0.1, 0.11], 'n_estimators' : [140]}]
gridSearch = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs= -1)
gridSearch.fit(X_train, y_train)
max_result = gridSearch.best_score_
best_para = gridSearch.best_params_