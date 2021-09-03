# Multiple Linear Regression or Linear Regression with multiple features (independent var)

import numpy as np
import pandas as pd


# Data Preprocessing
dataset = pd.read_csv('50_Startups.csv')

# reading the data
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# encoding the categorical variables ie. State

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEndcoder_X = LabelEncoder()
X[:,3] = labelEndcoder_X.fit_transform(X[:,3])

columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')

X= np.array(columnTransformer.fit_transform(X),dtype=np.float64)


# Avoiding the dummy variable Trap

X = X[:,1:]


# Splitting training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.api as smf

# we need to x0 = 1 to X because smf don't consider it like linear regression library
X = np.append(arr = np.ones((50,1)).astype(int),values=X,axis=1)
 
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = smf.OLS(y,X_opt).fit()
print(regressor_OLS.summary())

# remove x2 p(x2) = 0.99 > SL = 0.05

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = smf.OLS(y,X_opt).fit()
print(regressor_OLS.summary())

# remove x1 p(x1) = 0.94 > SL = 0.05


X_opt = X[:,[0,3,4,5]]
regressor_OLS = smf.OLS(y,X_opt).fit()
print(regressor_OLS.summary())


# remove x2 ie index 4 p(x2) = 0.602 > SL = 0.05

X_opt = X[:,[0,3,5]]
regressor_OLS = smf.OLS(y,X_opt).fit()
print(regressor_OLS.summary())


# remove x2 ie index 5 p(x2) = 0.06 > SL = 0.05

X_opt = X[:,[0,3]]
regressor_OLS = smf.OLS(y,X_opt).fit()
print(regressor_OLS.summary())




