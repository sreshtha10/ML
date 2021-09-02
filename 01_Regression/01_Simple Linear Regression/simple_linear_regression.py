# Simple Linear Regression ie LR with one variable

# Data Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values # experience
Y = dataset.iloc[:,1].values # salary

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)


# y = m*x + c  or y = Theta0 + Theta1*x

# dependent var = y
# independent var = x

# Here salary = y and x  = experience



# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the Test set results
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel("Years of Experience")
plt.ylabel('Salary')
plt.show()