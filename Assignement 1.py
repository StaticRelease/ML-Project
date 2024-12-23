from functools import reduce

from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

X,y=make_regression(n_samples=10000, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=10)
from sklearn.linear_model import LinearRegression

lr_m = LinearRegression()
lr_m.fit(X, y)

# takes the mean of the x and y values
mean_of_x = np.mean(X)
mean_of_y = np.mean(y)

# does the OLS formula for M
xy_num = np.sum((X.flatten() - mean_of_x) * (y - mean_of_y))
xy_den = np.sum((X.flatten() - mean_of_x) ** 2)

# OLS model formula
m = xy_num / xy_den
b = mean_of_y - m * mean_of_x
y_predict = m * X.flatten() + b


plt.scatter(X,y)
plt.plot(X.flatten(), y_predict, color = 'red')
plt.show()