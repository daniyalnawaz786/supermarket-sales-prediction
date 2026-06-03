from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# Paths (SAFE FIX)
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "super_market" / "supermarket_in.csv"
MODEL_PATH = BASE_DIR / "monthly_sales_model.pkl"

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv(DATA_PATH)
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# ===============================
# Monthly Aggregation
# ===============================
df["YearMonth"] = df["Order Date"].dt.to_period("M")
monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()

monthly_sales["YearMonth"] = monthly_sales["YearMonth"].dt.to_timestamp()
monthly_sales["Month"] = monthly_sales["YearMonth"].dt.month
monthly_sales["Year"] = monthly_sales["YearMonth"].dt.year

# ===============================
# Features
# ===============================
X = monthly_sales[["Month", "Year"]]
y = monthly_sales["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Model
# ===============================
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model trained!")
print("RMSE:", rmse)
print("R2:", r2)

# ===============================
# Save Model (SAFE PATH)
# ===============================
joblib.dump(model, MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)