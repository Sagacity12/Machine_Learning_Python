import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# %matplotlib inline

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)

# Optional exploration
# print(df.sample(5))
# print(df.describe())

# Correlation only on numeric columns to avoid string conversion errors
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print(corr_matrix)

feature_cols = ["ENGINESIZE", "FUELCONSUMPTION_COMB"]
target_col = "CO2EMISSIONS"
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

df = df[feature_cols + [target_col]]

# Extracting the input features and labels from the data set
X = df[feature_cols].to_numpy()
y = df[[target_col]].to_numpy()

# Preprocessing selected features
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)
pd.DataFrame(X_std).describe().round(2)

# 1. Create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Build a multiple linear reg model
regression = linear_model.LinearRegression()

# train the model in the training data
regression.fit(X_train, y_train)

# Print the coefficients
print("Coefficients:", regression.coef_)
print("Intercept:", regression.intercept_)

# get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares can be calculated relative to the original, unstandardized feature space as:
coef_original = regression.coef_ / std_devs_
intercept_original = regression.intercept_ - np.sum((means_ * coef_original) / std_devs_)
print("Original Coefficients:", coef_original)
print("Original Intercept:", intercept_original)

# Visualize model output
# Ensure X1, X2 and y_test have compatible shapes for 3D plotting
X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

# Create a mesh grid for plotting the regression plane
x1_surf, x2_surf = np.meshgrid(
    np.linspace(X1.min(), X1.max(), 100),
    np.linspace(X2.min(), X2.max(), 100),
)

y_surf = intercept_original + coef_original[0, 0] * x1_surf + coef_original[0, 1] * x2_surf

# Predict y values using train regression models to compare with actual y_test for above or below plane colors
y_pred = regression.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regression.predict(X_test)
above_plane = (y_test >= y_pred)[:, 0]
below_plane = (y_test < y_pred)[:, 0]

# Plotting
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection="3d")

# Plotting the data points above and below the plane in different colors
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane], label="Above Plane", s=70, alpha=0.7, ec="k")
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane], label="Below Plane", s=50, alpha=0.3, ec="k")

# Plotting the regression plane
ax.plot_surface(x1_surf, x2_surf, y_surf, color="k", alpha=0.21)

# Set the views and labels
ax.view_init(elev=10)

ax.legend(fontsize="x-large", loc="upper center")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel("ENGINESIZE", fontsize="xx-large")
ax.set_ylabel("FUELCONSUMPTION", fontsize="xx-large")
ax.set_zlabel("CO2 Emission", fontsize="xx-large")
ax.set_title("Multiple linear Regression of CO2 Emissions", fontsize="xx-large")
plt.tight_layout()
plt.show()