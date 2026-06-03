import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# ===============================
# Paths
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "super_market" / "supermarket_in.csv"
MODEL_PATH = BASE_DIR / "monthly_sales_model.pkl"

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Supermart Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return str(e)

df = load_data()
model = load_model()

# ===============================
# Sidebar
# ===============================
st.sidebar.markdown("### 📊 Navigation")
option = st.sidebar.radio(
    "Select:",
    ["📈 Dashboard Overview", "🎯 Product Recommendations", "📊 Sales Forecast", "🔍 Analytics"]
)

# ===============================
# Date Filter
# ===============================
min_date = df["Order Date"].min()
max_date = df["Order Date"].max()

date_range = st.sidebar.slider(
    "Date Range",
    min_value=min_date.date(),
    max_value=max_date.date(),
    value=(min_date.date(), max_date.date())
)

filtered_df = df[
    (df["Order Date"].dt.date >= date_range[0]) &
    (df["Order Date"].dt.date <= date_range[1])
]

# ===============================
# 🔥 SAFE EMPTY STATE (DO NOT BREAK APP)
# ===============================
if filtered_df.empty:
    st.warning("⚠️ No data for selected date range")
    st.info("👉 Adjust the date filter from sidebar")
    st.stop()

# ===============================
# DASHBOARD
# ===============================
if option == "📈 Dashboard Overview":

    st.title("🚀 Supermart Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")

    with col2:
        st.metric("Avg Order", f"${filtered_df['Sales'].mean():,.2f}")

    with col3:
        st.metric("Orders", len(filtered_df))

    with col4:
        st.metric("Products", filtered_df["Product Name"].nunique())

# ===============================
# PRODUCT RECOMMENDATIONS (FIXED FLOW)
# ===============================
elif option == "🎯 Product Recommendations":

    st.title("🎯 Product Recommendations")

    categories = filtered_df["Category"].dropna().unique()

    if len(categories) == 0:
        st.warning("No category data available")
    else:
        category = st.selectbox("Select Category:", categories)

        filtered_category = filtered_df[filtered_df["Category"] == category]

        if filtered_category.empty:
            st.warning("No products in this category")
        else:

            top_products = (
                filtered_category.groupby("Product Name")["Sales"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            st.bar_chart(top_products)

# ===============================
# SALES FORECAST (FIXED FLOW)
# ===============================
elif option == "📊 Sales Forecast":

    st.title("📊 Sales Forecast")

    if isinstance(model, str):
        st.error(f"Failed to load model: {model}")
    elif model:

        df_local = filtered_df.copy()

        df_local["YearMonth"] = df_local["Order Date"].dt.to_period("M")
        monthly_sales = df_local.groupby("YearMonth")["Sales"].sum().reset_index()
        monthly_sales["YearMonth"] = monthly_sales["YearMonth"].dt.to_timestamp()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_sales["YearMonth"],
            y=monthly_sales["Sales"],
            mode="lines+markers"
        ))
        st.plotly_chart(fig, use_container_width=True)

        today = datetime.date.today()
        next_month = today.month + 1 if today.month < 12 else 1
        next_year = today.year if today.month < 12 else today.year + 1

        input_data = pd.DataFrame({
            "Month": [next_month],
            "Year": [next_year]
        })

        prediction = model.predict(input_data)[0]

        st.metric("Next Month Prediction", f"${prediction:,.2f}")

    else:
        st.warning("Model not found")

# ===============================
# ANALYTICS (STABLE)
# ===============================
elif option == "🔍 Analytics":

    st.title("🔍 Analytics")

    st.write("Advanced analytics section")

    st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
    st.metric("Total Orders", len(filtered_df))