import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#  %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv" 

df=pd.read_csv(url)

df.sample(5)

df.describe() 

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']] 

viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 

viz.hist()

plt.show() 

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission") 
plt.show() 

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='Red') 
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.xlim(0, 27) 
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='Green') 
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

x = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy() 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)










# Plotting regression model results 

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.coef_[0] * X_train + regressor.intercept_, '-r') 
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# Selection of Fuel consumption feature from the dataframe and split the data 80%/20% into training and testing sets.

X = cdf.FUELCONSUMPTION_COMB.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear model

regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)


# Using the model to make test predictions on the fuel consumption testing data. 

y_pred = regressor.predict(X_test.reshape(-1, 1))

# Calculating the accuracy of the model using different evaluation metrics 

print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred)) 