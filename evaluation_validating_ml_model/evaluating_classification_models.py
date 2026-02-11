import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast cancer dataset
data = load_breast_cancer
X, y = data.data, data.target
labels = data.target_names
features = data.feature_names

print(data.DESCR)

print(data.target_names)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add Gaussian noise to the data
np.random.seed(42)
noise_factors = 0.5
X_noisy = X_scaled + noise_factors * np.random.normal(loc=0.0, scale=1.0, size=X_scaled.shape)

# Load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled, columns='feature_names')
df_noisy = pd.DataFrame(X_noisy, columns='feature_names')