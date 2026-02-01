import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from regression_tree import correlation_value

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()

# Data Visualization and Exploratory Analysis
df['custcat'].value_counts()

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
correlation_value = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
correlation_value

X = df.drop('custcat', axis=1)
y = df['custcat']

# Normalize Data
X_norm = StandardScaler().fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# KNN Classification
k = 3
# Training Model and Predict
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train, y_train)

# Predicting
yhat = knn_model.predict(X_test)

# Accuracy evaluation
accuracy = accuracy_score(y_test, yhat)
print(f"Accuracy: {accuracy:.2f}")
