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

