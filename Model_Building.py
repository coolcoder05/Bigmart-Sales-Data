

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train_modified.csv')
dataset.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
X = dataset.iloc[:,[0, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] ].values
y = dataset.iloc[:, 1].values


import statsmodels.formula.api as sm
X = np.append(arr = np.ones((8523, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [ 1, 4,8,9,10,11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,29,30]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_opt, y)

testset =  pd.read_csv('test_modified.csv')
testset.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
test_opt = testset.iloc[:,[0,3,7,8,9,10,11,12,13,14,15,16,20,21,22,23,24,25,26,27,28,29] ].values
#test_opt = np.append(arr = np.ones((5681, 1)).astype(int), values = test_opt, axis = 1)
y_pred = regressor.predict(test_opt)

