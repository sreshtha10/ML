#Non linear non continuous Decision Tree Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])


# Visualizing the regression results (for higher res and smoother curve)
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.show()