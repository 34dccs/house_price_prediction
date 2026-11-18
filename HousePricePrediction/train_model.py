
import pandas as pd # type: ignore
import numpy as np # pyright: ignore[reportMissingImports]
import joblib # type: ignore
import os

from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import OneHotEncoder # pyright: ignore[reportMissingModuleSource]
from sklearn.pipeline import Pipeline # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error # pyright: ignore[reportMissingModuleSource]

# Generate sample dataset if CSV does not exist
if not os.path.exists("house_data.csv"):

    np.random.seed(42)

    locations = ["Delhi", "Mumbai", "Bangalore",
                 "Chandigarh", "Panipat", "Ambala"]

    data = []

    location_rates = {
        "Delhi": 9000,
        "Mumbai": 12000,
        "Bangalore": 7500,
        "Chandigarh": 8000,
        "Panipat": 3500,
        "Ambala": 3000
    }

    for i in range(1000):
        area = np.random.randint(500, 4000)
        bedrooms = np.random.randint(1, 6)
        bathrooms = np.random.randint(1, 5)
        age = np.random.randint(0, 30)
        location = np.random.choice(locations)

        price = (
            area * location_rates[location]
            + bedrooms * 300000
            + bathrooms * 200000
            - age * 30000
            + np.random.randint(-300000, 300000)
        )

        data.append([
            area, bedrooms, bathrooms,
            age, location, max(price, 500000)
        ])

    df = pd.DataFrame(data, columns=[
        "Area", "Bedrooms", "Bathrooms",
        "Age", "Location", "Price"
    ])

    df.to_csv("house_data.csv", index=False)

    print("Dataset created successfully!")

# Load dataset
df = pd.read_csv("house_data.csv")

X = df[["Area", "Bedrooms", "Bathrooms", "Age", "Location"]]
y = df["Price"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"),
         ["Location"])
    ],
    remainder="passthrough"
)

# Create ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Model Training Completed!")
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:",
      mean_absolute_error(y_test, predictions))

# Save model
joblib.dump(model, "house_model.pkl")

print("Model saved successfully!")
