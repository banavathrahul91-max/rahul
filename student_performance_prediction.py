"""
Student Performance Prediction System

This project demonstrates:
1. Data preprocessing
2. Exploratory Data Analysis (EDA)
3. Logistic Regression for Pass/Fail prediction
4. Linear Regression for marks prediction
5. Feature scaling and model evaluation

Expected input file: student_data.csv

Recommended columns:
- Hours_Studied
- Attendance
- Assignment_Score
- Sleep_Hours
- Study_Sessions
- Marks
- Optional categorical columns such as Gender

If student_data.csv does not exist, a sample dataset is generated automatically
so that the project can be executed immediately.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path("student_data.csv")
RANDOM_STATE = 42


def create_sample_dataset(rows: int = 200) -> pd.DataFrame:
    """Create a reproducible sample dataset for demonstration."""
    rng = np.random.default_rng(RANDOM_STATE)

    hours_studied = rng.uniform(1, 10, rows)
    attendance = rng.uniform(50, 100, rows)
    assignment_score = rng.uniform(35, 100, rows)
    sleep_hours = rng.uniform(4, 9, rows)
    study_sessions = rng.integers(1, 10, rows)

    marks = (
        2.8 * hours_studied
        + 0.35 * attendance
        + 0.38 * assignment_score
        + 1.2 * sleep_hours
        + 1.4 * study_sessions
        + rng.normal(0, 5, rows)
    )
    marks = np.clip(marks, 0, 100)

    return pd.DataFrame(
        {
            "Hours_Studied": hours_studied.round(2),
            "Attendance": attendance.round(2),
            "Assignment_Score": assignment_score.round(2),
            "Sleep_Hours": sleep_hours.round(2),
            "Study_Sessions": study_sessions,
            "Marks": marks.round(2),
        }
    )


def load_and_preprocess_data() -> pd.DataFrame:
    """Load, clean, and prepare the student data."""
    if DATA_FILE.exists():
        data = pd.read_csv(DATA_FILE)
        print(f"Loaded dataset from {DATA_FILE}")
    else:
        data = create_sample_dataset()
        data.to_csv(DATA_FILE, index=False)
        print(f"{DATA_FILE} was not found. Created a sample dataset.")

    data = data.drop_duplicates().reset_index(drop=True)

    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    categorical_columns = data.select_dtypes(include="object").columns
    for column in categorical_columns:
        data[column] = data[column].fillna(data[column].mode().iloc[0])

    if "Marks" not in data.columns:
        raise ValueError("The dataset must contain a 'Marks' column.")

    # Pass mark is set to 40. Change this threshold if required.
    data["Pass_Fail"] = (data["Marks"] >= 40).astype(int)

    # Convert categorical features to numeric dummy variables.
    data = pd.get_dummies(data, columns=list(categorical_columns), drop_first=True)
    return data


def perform_eda(data: pd.DataFrame) -> None:
    """Display basic EDA charts."""
    print("\nDataset shape:", data.shape)
    print("\nFirst five rows:\n", data.head())
    print("\nDescriptive statistics:\n", data.describe())
    print("\nMissing values:\n", data.isnull().sum())

    numeric_data = data.select_dtypes(include=np.number)

    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Student Performance Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    if "Hours_Studied" in data.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=data, x="Hours_Studied", y="Marks", hue="Pass_Fail")
        plt.title("Marks vs Hours Studied")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(7, 5))
    sns.histplot(data=data, x="Marks", hue="Pass_Fail", kde=True)
    plt.title("Marks Distribution")
    plt.tight_layout()
    plt.show()


def train_models(data: pd.DataFrame) -> None:
    """Train and evaluate classification and regression models."""
    feature_columns = [column for column in data.columns if column not in {"Marks", "Pass_Fail"}]
    X = data[feature_columns]

    # Logistic Regression: predict pass or fail.
    y_classification = data["Pass_Fail"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X,
        y_classification,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_classification,
    )

    classification_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    classification_model.fit(X_train_c, y_train_c)
    predictions_c = classification_model.predict(X_test_c)

    print("\n=== Logistic Regression: Pass/Fail Prediction ===")
    print(f"Accuracy: {accuracy_score(y_test_c, predictions_c):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test_c, predictions_c))
    print("Classification report:\n", classification_report(y_test_c, predictions_c))

    # Linear Regression: estimate marks.
    y_regression = data["Marks"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_regression, test_size=0.2, random_state=RANDOM_STATE
    )

    regression_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    regression_model.fit(X_train_r, y_train_r)
    predictions_r = regression_model.predict(X_test_r)

    print("\n=== Linear Regression: Marks Prediction ===")
    print(f"MAE: {mean_absolute_error(y_test_r, predictions_r):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, predictions_r)):.3f}")
    print(f"R² Score: {r2_score(y_test_r, predictions_r):.3f}")

    # Show one example prediction.
    sample = X_test_c.iloc[[0]]
    predicted_status = classification_model.predict(sample)[0]
    predicted_marks = regression_model.predict(sample)[0]
    print("\nExample prediction:")
    print("Predicted result:", "Pass" if predicted_status == 1 else "Fail")
    print(f"Predicted marks: {predicted_marks:.2f}")


def main() -> None:
    data = load_and_preprocess_data()
    perform_eda(data)
    train_models(data)


if __name__ == "__main__":
    main()
