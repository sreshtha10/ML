# Polynomial Linear Regression


# It is called linear too because in Linear regression we don't
# talk about functional relationship b/w y and x. Instead we are
# talking about y and all the coefficients.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

# We can see there is a Polynomial relationship b/w Level and salary
plt.scatter(dataset.iloc[:,1].values,dataset.iloc[:,2].values)
plt.ylabel("Salary")
plt.xlabel("Level")
plt.show()


X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# No splitting because our dataset is very small. Also we want very accurate results

# Fitting the Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X,y)



# Fitting Polynomial Regression to the dataset.

# we converted x to a feature matrix x_poly with features in polynomial form.
from sklearn.preprocessing import PolynomialFeatures
polyRegressor = PolynomialFeatures(degree=5)
X_poly = polyRegressor.fit_transform(X)

# fitting x_poly in linear regression model
linearRegressor2 = LinearRegression()
linearRegressor2.fit(X_poly, y)


# Visualizing the Linear Regression results
y_pred = linearRegressor.predict(X)
plt.scatter(X,y,color='red')
plt.plot(X,y_pred,color='blue')
plt.ylabel("Salary")
plt.xlabel("Level")
plt.title('Results of Linear Regression Model')
plt.show()



# Visualizing the Polynomial Linear Regression results


y_pred2 = linearRegressor2.predict(polyRegressor.fit_transform(X))
plt.scatter(X,y,color='red')
plt.plot(X,y_pred2,color='blue')
plt.ylabel("Salary")
plt.xlabel("Level")
plt.title('Results of Pol ynomial Linear Regression Model')
plt.show()



# Prediciting the new result with Linear Regression Model

sal = [[6.5]]
print(linearRegressor.predict(sal))


# Predicting the new result with Polynomial Linear Regression Model


print(linearRegressor2.predict(polyRegressor.fit_transform(sal)))




