# ============================================================
# STEP 1: Import Libraries
# ============================================================
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# STEP 2: Load Dataset
# The Diabetes dataset is available on Kaggle.
# Columns: Pregnancies, Glucose, BloodPressure, SkinThickness,
#          Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
# ============================================================

column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head())

# ============================================================
# STEP 3: Split Features and Target
# ============================================================
X = df.drop("Outcome", axis=1)   # Features (8 columns)
y = df["Outcome"]                 # Target: 0 = No Diabetes, 1 = Diabetes

# ============================================================
# STEP 4: Train-Test Split (80% train, 20% test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# STEP 5: Feature Scaling (StandardScaler)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# STEP 6: Train Logistic Regression Model
# ============================================================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ============================================================
# STEP 7: Evaluate the Model
# ============================================================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# ============================================================
# STEP 8: Save the Model and Scaler as .pkl files
# ============================================================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ model.pkl and scaler.pkl saved successfully!")
