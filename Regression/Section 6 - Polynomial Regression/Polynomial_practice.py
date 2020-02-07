# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:00:34 2019

@author: imran
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
# [: 1] will take the X as a vector. but we need X as a matrix. so [:, 1:2]
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''very few dataset. no need to split.
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8, random_state = 0)'''

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y);



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(4)
X_poly = poly_reg.fit_transform(X)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_poly, y)


plt.scatter(X, y, color='red')
plt.plot(X, linear_reg.predict(X), color='blue')
plt.title('Linear Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


plt.scatter(X, y, color='green')
plt.plot(X, linear_reg_2.predict(poly_reg.fit_transform(X)), color='yellow')
plt.title('Plynomial Regression Model')
plt.xlabel('Levels of Position')
plt.ylabel('Salary')
plt.show()


#prediction
linear_reg.predict(np.array([6.5]).reshape(1, 1))
linear_reg_2.predict(poly_reg.fit_transform([[6.5]]))