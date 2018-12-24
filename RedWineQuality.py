                  #Red Wine Quality Prediction Using Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality-red.csv')
X = dataset.iloc[:, :-1].values # Independent Variables
y = dataset.iloc[:, 11].values   # Dependent Variables

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, :-1])
X[:, :-1] = imputer.transform(X[:, :-1])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into Training Set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#Fitting Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting The Test Set Result
y_pred = regressor.predict(X_test)
y_p = y_pred.round(decimals=0)


from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(y_test,y_p)
ac = accuracy_score(y_test,y_p)





