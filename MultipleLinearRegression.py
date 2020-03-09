# Multiple Linear Regression

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Importing the dataset

ds = pd.read_csv('50_Startups.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# There is no missing data

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
ohe_X = OneHotEncoder(categorical_features=[-1])
X[:,-1] = le_X.fit_transform(X[:,-1])
X = ohe_X.fit_transform(X).toarray()

# Avoiding the dummy variable trap

X = X[:,1:]

# Feature Scaling is not required as the linear model library takes care of this for us

# Backwards elimination

import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# New York has the highest p-value (>0.05) so that variable as been removed, we will run Reg_OLS and review the metrics again

X_opt = X[:,[0,1,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# Florida has the highest p-value (>0.05) so that variable as been removed, we will run Reg_OLS and review the metrics again

X_opt = X[:,[0,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# Administration has the highest p-value (>0.05) so that variable as been removed, we will run Reg_OLS and review the metrics again

X_opt = X[:,[0,3,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# Marketing spend has the highest p-value (>0.05)  so that variable as been removed, we will run Reg_OLS and review the metrics again

X_opt = X[:,[0,3]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# As the adjusted R-squared has decreased, we will add the "Marketing Spend" variable back and declare this our multiple linear regression model
# We conclude that we should invest in companies that are investing highly in "R&D" (Research and Development), with "Marketing Spend" being the
# additional (but weaker) contributing factor

X_opt = X[:,[0,3,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# Importing our dataset

X = ds.iloc[:,[0,2]].values
y = ds.iloc[:,-1].values

# Splitting the dataset into the training set and testing set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the multiple linear regression to the dataset

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the test set values

y_pred = lr.predict(X_test)

# Calculating the Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
# MSE is more popular than MAE because MSE "punishes" larger errors. But, RMSE is even more popular than MSE because 
# RMSE is interpretable in the "y" units.

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 