import pandas as pd
import numpy as np
data = pd.read_csv('C:\Misty_ict\winequality-red.csv')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# Initialize the model

model = RandomForestClassifier(n_estimators=100, random_state=42)

df = data.drop(columns=["residual sugar", "free sulfur dioxide", "citric acid"])

X = df.drop('quality', axis=1)
y = df['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

import pickle
with open("model.pkl","wb") as f:
  pickle.dump(model,f)