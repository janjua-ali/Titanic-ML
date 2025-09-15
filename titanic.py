"""
Titanic Survival Prediction - Internship Portfolio Project
Author: Ali Noor
Email: an2345001@gmail.com

This project demonstrates:
- Data preprocessing
- Feature engineering
- Logistic Regression model
- Evaluation (accuracy, confusion matrix, classification report)
- Cross-validation
- Basic visualization

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("data/titanic.csv")
print("Dataset loaded successfully. Shape:", df.shape)

# ------------------------------
# 2. Data Preprocessing
# ------------------------------
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[features + ["Survived"]]

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# ------------------------------
# 3. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 4. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# 5. Model Training
# ------------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ------------------------------
# 6. Predictions & Evaluation
# ------------------------------
y_pred = model.predict(X_test)

print("\n📊 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("\nCross-validation Accuracy:", cv_scores.mean())

# ------------------------------
# 7. Visualization
# ------------------------------
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
