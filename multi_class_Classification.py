import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

# Resolve the target column name once to avoid mismatches
target_col = "NObeyesdad"
if target_col not in data.columns:
    alt_name = "NObeyesdaa"
    if alt_name in data.columns:
        target_col = alt_name
    else:
        raise ValueError(f"Target column not found. Available columns: {list(data.columns)}")

# Exploratory Data Analysis
sns.countplot(data=data, y=target_col)
plt.title('Distribution of Obesity Levels')
plt.show()

# Preprocessing the data
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])

scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# One-hot encoding
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
if target_col in categorical_columns:
    categorical_columns.remove(target_col)

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

encode_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encode_df], axis=1)

# Encode the target variable
prepped_data[target_col] = prepped_data[target_col].astype('category').cat.codes

# Separate the input and target data
X = prepped_data.drop(target_col, axis=1)
y = prepped_data[target_col]

# Splitting the data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Training logistic regression model using One-vs-All
model_ova = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_ova.fit(X_train, y_train)

# Predictions
y_prep_ova = model_ova.predict(X_test)

# Evaluation metrics for OVA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_prep_ova), 2)}%")

# Training logistic regression model using One-vs-One
# model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
# model_ovo.fit(X_train, y_train)

# Predictions
# y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
# print("One-vs-One (OvO) Strategy")
# print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")
