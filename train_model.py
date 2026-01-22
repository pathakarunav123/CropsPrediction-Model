# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json

# =============================
# Load Data
# =============================
df1 = pd.read_csv("Crops_data_ultra_cleaned.csv")

# =============================
# Define crops and their columns
# =============================
crop_columns = {
    "Rice": {
        "area": "RICE AREA (1000 ha)",
        "production": "RICE PRODUCTION (1000 tons)",
        "yield": "RICE YIELD (Kg per ha)"
    },
    "Wheat": {
        "area": "WHEAT AREA (1000 ha)",
        "production": "WHEAT PRODUCTION (1000 tons)",
        "yield": "WHEAT YIELD (Kg per ha)"
    },
    "Maize": {
        "area": "MAIZE AREA (1000 ha)",
        "production": "MAIZE PRODUCTION (1000 tons)",
        "yield": "MAIZE YIELD (Kg per ha)"
    },
    "Cotton": {
        "area": "COTTON AREA (1000 ha)",
        "production": "COTTON PRODUCTION (1000 tons)",
        "yield": "COTTON YIELD (Kg per ha)"
    },
    "Sugarcane": {
        "area": "SUGARCANE AREA (1000 ha)",
        "production": "SUGARCANE PRODUCTION (1000 tons)",
        "yield": "SUGARCANE YIELD (Kg per ha)"
    }
}

# =============================
# Model Training
# =============================
rf_params = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [5, 10, 15, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", None]
}

crop_stats = {}

for crop, cols in crop_columns.items():
    print(f"\n🌾 Training model for {crop}...")

    # Build dataset for this crop
    if not all(c in df1.columns for c in cols.values()):
        print(f"⚠️ Skipping {crop}, missing columns in dataset.")
        continue

    crop_df = df1[[cols["area"], cols["production"], cols["yield"]]].dropna()

    if crop_df.shape[0] < 20:
        print(f"⚠️ Skipping {crop}, not enough data ({crop_df.shape[0]} rows).")
        continue

    X = crop_df[[cols["area"], cols["production"]]]
    y = crop_df[cols["yield"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=42))
    ])

    search = RandomizedSearchCV(
        rf_pipeline, rf_params, n_iter=10,
        scoring="r2", cv=3, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    print(f"✅ {crop} - R²: {r2_score(y_test, y_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    # Save yield model
    joblib.dump(search.best_estimator_, f"{crop.lower()}_yield_model.pkl")
    print(f"💾 Saved: {crop.lower()}_yield_model.pkl")

    # Save production model (train separately: area + yield → production)
    X_prod = crop_df[[cols["area"], cols["yield"]]]
    y_prod = crop_df[cols["production"]]

    search_prod = RandomizedSearchCV(
        rf_pipeline, rf_params, n_iter=10,
        scoring="r2", cv=3, random_state=42, n_jobs=-1
    )
    search_prod.fit(X_prod, y_prod)

    joblib.dump(search_prod.best_estimator_, f"{crop.lower()}_prod_model.pkl")
    print(f"💾 Saved: {crop.lower()}_prod_model.pkl")

    # Save yield stats
    crop_stats[crop] = {
        "min_yield": float(y.min()),
        "max_yield": float(y.max()),
        "avg_yield": float(y.mean())
    }

# Save stats JSON
with open("crop_yield_stats.json", "w") as f:
    json.dump(crop_stats, f, indent=4)

print("\n📊 Saved crop_yield_stats.json")
