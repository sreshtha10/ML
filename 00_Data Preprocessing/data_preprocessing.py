# Data Preprocessing Template

# importing libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv('Data.csv')


# create matrix X -> matrix of features (dependent var)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


# replace the missing data by the mean of the values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


# handling categorical variables. In this case Purchased and country col.

# we have to encode these into numbers.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X= LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0]) 

# The problem with this is that the ML model will think this data in terms of values ie 2>1 etc.
# but actually these are just categories.

columnTansformer = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(columnTansformer.fit_transform(X),dtype=np.float64)


labelEncoder_Y = LabelEncoder()
Y[:] = labelEncoder_Y.fit_transform(Y[:])


# splitting data into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# Feature Scaling is imp because a lot of ml models are based on Euclidian Distance and alogrithm will converge much faster

# 1st way standardisation xs = x- mean(x) divided by std deviation(x)
# 2nd way normalization xnorm = (x-min(x))/max(x)-min(x)

from sklearn.preprocessing import StandardScaler 

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


