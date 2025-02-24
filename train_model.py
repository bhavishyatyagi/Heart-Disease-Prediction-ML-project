import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("heart.csv")

# Convert categorical variables into numerical format
df = pd.get_dummies(df, drop_first=True)

# Save feature names
feature_names = df.drop("HeartDisease", axis=1).columns

# Split data into features and labels
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model, scaler, and feature names
joblib.dump(model, "logistic_regression.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(feature_names, "feature_names.joblib")

print("✅ Model, scaler, and feature names saved successfully!")
