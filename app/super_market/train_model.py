# ===============================
# Supermart Monthly Sales Forecast Model
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv("app/super_market/supermarket(in).csv")

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# ===============================
# 2️⃣ Monthly Sales Aggregation
# ===============================
df["YearMonth"] = df["Order Date"].dt.to_period("M")

monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()

# Convert period to datetime
monthly_sales["YearMonth"] = monthly_sales["YearMonth"].dt.to_timestamp()

# Extract time features
monthly_sales["Month"] = monthly_sales["YearMonth"].dt.month
monthly_sales["Year"] = monthly_sales["YearMonth"].dt.year

# ===============================
# 3️⃣ Features + Target
# ===============================
X = monthly_sales[["Month", "Year"]]
y = monthly_sales["Sales"]

# ===============================
# 4️⃣ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5️⃣ Train Model
# ===============================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6️⃣ Evaluation
# ===============================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Monthly Sales Forecast Model Trained!")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# ===============================
# 7️⃣ Save Model
# ===============================
joblib.dump(model, "monthly_sales_model.pkl")

print("✅ Model Saved as monthly_sales_model.pkl")
