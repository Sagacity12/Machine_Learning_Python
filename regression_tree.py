from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

import warnings

from multiple_reg import X_test

warnings.filterwarnings('ignore')

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data.head()

correlation_value = raw_data.corr()[ 'tip_amount' ].drop('tip_amount')
correlation_value.plot(kind='barh', figsize=(10, 8), title='Correlation with Tip Amount', xlabel='Correlation', ylabel='Features', grid=True)

# Datasets Preprocessing
# extract the labels from the dataframe
y=- raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l2', copy=False)

# Data Trian / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a Decision Tree regression model with Scikit-learn
dt_regressor = DecisionTreeRegressor(criterion= 'squared_error', max_depth=8, random_state=35)
dt_regressor.fit(X_train, y_train)

# Evaluate the Scikit-Learn and Snap ML Decision Tree regression models
y_pred_dt = dt_regressor.predict(X_test)
# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred_dt)
print("Scikit-Learn Decision Tree MSE: {0:.2f}".format(mse_score))

r2_score = dt_regressor.score(X_test, y_test)
print('R^2 score : {0: .3f}'.format(r2_score))


