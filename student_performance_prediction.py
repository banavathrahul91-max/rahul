# student_performance_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
 1. Load dataset
df = pd.read_csv("student_data.csv")
print("Dataset shape:", df.shape)
print(df.head())
 2. Data cleaning and preprocessing
 Check missing values
print("\nMissing values:\n", df.isnull().sum())
# Drop duplicate rows
df = df.drop_duplicates().reset_index(drop=True)
# Fill numeric missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
# Fill categorical missing values
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert target variable to binary classification if needed
if "Marks" in df.columns and "Pass_Fail" not in df.columns:
    df["Pass_Fail"] = (df["Marks"] >= 40).astype(int)

# Encode categorical columns
if len(categorical_cols) > 0:
    df = pd.get_dummies(df, columns=list(categorical_cols), drop_first=True)

print("\nProcessed dataset columns:\n", df.columns.tolist())
print(df.head())

 3. Exploratory Data Analysis (EDA)
# Summary statistics
print("\nDescriptive statistics:\n", df.describe())

# Correlation heatmap
numeric_features = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Distribution plots
for col in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

# Scatter plot: Marks vs Hours Studied
if "Marks" in df.columns and "Hours_Studied" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="Hours_Studied", y="Marks")
    plt.title("Marks vs Hours Studied")
    plt.tight_layout()
    plt.show()

 4. Split data into training and testing sets
# For classification (Pass/Fail)
if "Pass_Fail" in df.columns:
    X_class = df.drop(
        columns=["Pass_Fail", "Marks"] if "Marks" in df.columns else ["Pass_Fail"]
    )
    y_class = df["Pass_Fail"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

# For regression (Marks)
if "Marks" in df.columns:
    X_reg = df.drop( columns=["Marks", "Pass_Fail"] if "Pass_Fail" in df.columns else ["Marks"] )
    y_reg = df["Marks"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42 )

5. Train and evaluate models
# 5A. Logistic Regression (Classification)
if "Pass_Fail" in df.columns:
    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    clf_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = clf_pipeline.predict(X_test_c)
    accuracy = accuracy_score(y_test_c, y_pred_c)
    cm = confusion_matrix(y_test_c, y_pred_c)
    report = classification_report(y_test_c, y_pred_c)
    print("\n=== Logistic Regression Evaluation ===")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

 5B. Linear Regression (Regression)
if "Marks" in df.columns:
    reg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    reg_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = reg_pipeline.predict(X_test_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    print("\n=== Linear Regression Evaluation ===")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² Score:", r2)

 6. Prediction examples
# Example for a new student
if "Pass_Fail" in df.columns:
    sample = X_test_c.iloc[[0]]
    predicted_class = clf_pipeline.predict(sample)
    print("\nSample prediction for Pass/Fail:", predicted_class[0])

if "Marks" in df.columns:
    sample_marks = X_test_r.iloc[[0]]
    predicted_marks = reg_pipeline.predict(sample_marks)
    print("Sample prediction for Marks:", predicted_marks[0])
