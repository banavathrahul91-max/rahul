
House Price Prediction System

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("house_data.csv")
RANDOM_STATE = 42
TARGET_COLUMN = "Price"
def create_sample_dataset(rows: int = 250) -> pd.DataFrame:
    """Create a reproducible sample house-price dataset."""
    rng = np.random.default_rng(RANDOM_STATE)
    locations = np.array(["City Center", "Suburban", "Outskirts", "Near Metro"])
    area = rng.integers(500, 3500, rows)
    bedrooms = rng.integers(1, 6, rows)
    bathrooms = np.maximum(1, bedrooms - rng.integers(0, 2, rows))
    age = rng.integers(0, 31, rows)
    location = rng.choice(locations, rows)
    location_bonus = {
        "City Center": 180000,
        "Near Metro": 130000,
        "Suburban": 70000,
        "Outskirts": 20000,
    }
    price = (
        850 * area
        + 45000 * bedrooms
        + 30000 * bathrooms
        - 3500 * age
        + np.array([location_bonus[item] for item in location])
        + rng.normal(0, 65000, rows)
    )
    return pd.DataFrame(
        {
            "Area_sqft": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age_years": age,
            "Location": location,
            "Price": np.maximum(price, 50000).round(2),
        }
    )

def load_data() -> pd.DataFrame:
    """Load the dataset or create a sample dataset if needed."""
    if DATA_FILE.exists():
        data = pd.read_csv(DATA_FILE)
        print(f"Loaded dataset from {DATA_FILE}")
    else:
        data = create_sample_dataset()
        data.to_csv(DATA_FILE, index=False)
        print(f"{DATA_FILE} was not found. Created a sample dataset.")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"The dataset must contain a '{TARGET_COLUMN}' column.")
return data.drop_duplicates().reset_index(drop=True)

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived features when the required source columns exist."""
    data = data.copy()
    if {"Area_sqft", "Bedrooms"}.issubset(data.columns):
        data["Area_per_Bedroom"] = data["Area_sqft"] / data["Bedrooms"].replace(0, np.nan)
    if {"Bathrooms", "Bedrooms"}.issubset(data.columns):
        data["Bathrooms_per_Bedroom"] = data["Bathrooms"] / data["Bedrooms"].replace(0, np.nan)
    return data

def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = features.select_dtypes(exclude=np.number).columns.tolist()
 numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

def main() -> None:
    data = add_features(load_data())
    print("\nDataset shape:", data.shape)
    print("\nFirst five rows:\n", data.head())
    print("\nMissing values before preprocessing:\n", data.isnull().sum())
    print("\nDescriptive statistics:\n", data.describe(include="all"))
  features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
     Linear Regression is used as the main regression model.
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor(features)),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("\n=== House Price Prediction Results ===")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R² Score: {r2:.3f}")
    results = pd.DataFrame(
        {
            "Actual_Price": y_test.to_numpy(),
            "Predicted_Price": predictions,
        }
    ).reset_index(drop=True)
    print("\nActual versus predicted prices:\n", results.head(10))
    
Comparison graph required by the project objective.
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=results, x="Actual_Price", y="Predicted_Price", s=70)
    minimum = min(results["Actual_Price"].min(), results["Predicted_Price"].min())
    maximum = max(results["Actual_Price"].max(), results["Predicted_Price"].max())
    plt.plot([minimum, maximum], [minimum, maximum], "r--", label="Perfect prediction")
    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("house_price_comparison.png", dpi=150)
    plt.show()

     Predict one test house as an example.
    example_house = X_test.iloc[[0]]
    example_prediction = model.predict(example_house)[0]
    print(f"\nExample predicted house price: {example_prediction:,.2f}")

if __name__ == "__main__":
    main()
