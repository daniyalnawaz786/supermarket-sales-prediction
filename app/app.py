import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Supermart Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cosmic Theme
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #0A0E27 0%, #1B1F3B 50%, #0F1B2E 100%);
        color: #E0E0E0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B1F3B 0%, #0F1B2E 100%);
        border-right: 2px solid #9D4EDD;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #A78BFA;
        text-shadow: 0 0 20px rgba(157, 78, 221, 0.5);
        font-weight: 700;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background-color: rgba(157, 78, 221, 0.1);
        border: 2px solid #9D4EDD;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(157, 78, 221, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #9D4EDD 0%, #C77DFF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(157, 78, 221, 0.6);
        transform: translateY(-2px);
    }
    
    /* Select/Input Boxes */
    .stSelectbox, .stMultiSelect {
        border-radius: 8px;
    }
    
    /* Sidebar Content */
    [data-testid="stSidebarNav"] {
        color: #A78BFA;
    }
    
    /* Radio Buttons */
    [data-testid="stRadio"] label {
        color: #E0E0E0;
        font-weight: 500;
    }
    
    /* Success Messages */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22C55E;
        border-radius: 8px;
    }
    
    /* Info Messages */
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
    }
    
    /* Card Style */
    .card {
        background: linear-gradient(135deg, rgba(157, 78, 221, 0.1) 0%, rgba(199, 125, 255, 0.05) 100%);
        border: 2px solid #9D4EDD;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(157, 78, 221, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# Load Dataset & Model
# ===============================
@st.cache_resource
def load_data():
    df = pd.read_csv("super_market/supermarket_in.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load("monthly_sales_model.pkl")
    except:
        return None

df = load_data()
model = load_model()

# ===============================
# Header Section
# ===============================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🚀 SUPERMART ANALYTICS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9D4EDD; font-size: 16px;'>✨ Cosmic Dashboard for Sales Intelligence</p>", unsafe_allow_html=True)

st.divider()

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.markdown("### 📊 Navigation Menu")
option = st.sidebar.radio(
    "Select a Feature:",
    ["📈 Dashboard Overview", "🎯 Product Recommendations", "📊 Sales Forecast", "🔍 Analytics"],
    key="nav_option"
)

st.sidebar.divider()
st.sidebar.markdown("### 📅 Filter by Date Range")
min_date = df["Order Date"].min()
max_date = df["Order Date"].max()
date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=min_date.date(),
    max_value=max_date.date(),
    value=(min_date.date(), max_date.date())
)

filtered_df = df[(df["Order Date"].dt.date >= date_range[0]) & (df["Order Date"].dt.date <= date_range[1])]

# ===============================
# 1. DASHBOARD OVERVIEW
# ===============================
if option == "📈 Dashboard Overview":
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df["Sales"].sum()
        st.metric("💰 Total Sales", f"${total_sales:,.2f}", delta=f"${total_sales/100:,.2f}")
    
    with col2:
        avg_order = filtered_df["Sales"].mean()
        st.metric("📦 Avg Order Value", f"${avg_order:,.2f}")
    
    with col3:
        total_orders = len(filtered_df)
        st.metric("🛍️ Total Orders", f"{total_orders:,}")
    
    with col4:
        unique_products = filtered_df["Product Name"].nunique()
        st.metric("🏷️ Unique Products", f"{unique_products}")
    
    st.divider()
    
    # Sales by Category
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Sales by Category")
        category_sales = filtered_df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        fig1 = go.Figure(data=[
            go.Bar(x=category_sales.index, y=category_sales.values,
                   marker=dict(color=category_sales.values, colorscale='Purples'))
        ])
        fig1.update_layout(
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0'),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Sales Distribution")
        fig2 = go.Figure(data=[
            go.Pie(labels=filtered_df["Category"].unique(),
                   values=[filtered_df[filtered_df["Category"]==cat]["Sales"].sum() 
                          for cat in filtered_df["Category"].unique()],
                   marker=dict(colors=['#9D4EDD', '#C77DFF', '#E0AAFF', '#A78BFA']))
        ])
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0'),
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # Time Series Analysis
    st.markdown("#### 📊 Sales Trend Over Time")
    daily_sales = filtered_df.groupby(filtered_df["Order Date"].dt.date)["Sales"].sum()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=daily_sales.index, y=daily_sales.values,
        mode='lines+markers',
        name='Daily Sales',
        line=dict(color='#9D4EDD', width=3),
        marker=dict(size=6, color='#C77DFF')
    ))
    fig3.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)

# ===============================
# 2. PRODUCT RECOMMENDATIONS
# ===============================
elif option == "🎯 Product Recommendations":
    st.markdown("### 🎯 Intelligent Product Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category = st.selectbox("Select Category:", filtered_df["Category"].unique())
    
    with col2:
        top_n = st.slider("Show Top N Products:", 1, 20, 5)
    
    filtered_category = filtered_df[filtered_df["Category"] == category]
    top_products = filtered_category.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(top_n)
    
    st.markdown(f"#### 🔥 Top {top_n} Products in {category}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                y=top_products.index,
                x=top_products.values,
                orientation='h',
                marker=dict(color=top_products.values, colorscale='Viridis')
            )
        ])
        fig.update_layout(
            hovermode='y',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0'),
            xaxis_title='Total Sales ($)',
            yaxis_title='Product Name',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Stats")
        for idx, (product, sales) in enumerate(top_products.items(), 1):
            st.metric(f"#{idx}", f"${sales:,.2f}", product[:20])
    
    st.divider()
    
    # Product Details Table
    st.markdown("#### 📋 Detailed Product Information")
    product_table = filtered_category.groupby("Product Name").agg({
        "Sales": ["sum", "mean", "count"],
        "Quantity": "sum"
    }).round(2).sort_values(("Sales", "sum"), ascending=False).head(top_n)
    product_table.columns = ['Total Sales', 'Avg Sale', 'Orders', 'Total Quantity']
    st.dataframe(product_table, use_container_width=True)

# ===============================
# 3. SALES FORECAST
# ===============================
elif option == "📊 Sales Forecast":
    st.markdown("### 📈 Advanced Sales Forecasting")
    
    if model is not None:
        # Monthly Aggregation
        df["YearMonth"] = df["Order Date"].dt.to_period("M")
        monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()
        monthly_sales["YearMonth"] = monthly_sales["YearMonth"].dt.to_timestamp()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 Historical Monthly Sales Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_sales["YearMonth"],
                y=monthly_sales["Sales"],
                mode='lines+markers',
                name='Monthly Sales',
                line=dict(color='#9D4EDD', width=3),
                marker=dict(size=8, color='#C77DFF')
            ))
            fig.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0E0E0'),
                xaxis_title='Month',
                yaxis_title='Sales ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col1:
            st.markdown("#### 🔮 Next Month Prediction")
            today = datetime.date.today()
            next_month = today.month + 1 if today.month < 12 else 1
            next_year = today.year if today.month < 12 else today.year + 1
            
            input_data = pd.DataFrame({
                "Month": [next_month],
                "Year": [next_year]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    st.metric(
                        f"💰 Predicted Sales - {next_month}/{next_year}",
                        f"${prediction:,.2f}",
                        delta=f"Based on historical data"
                    )
                
                with col_pred2:
                    avg_monthly = monthly_sales["Sales"].mean()
                    change = ((prediction - avg_monthly) / avg_monthly * 100)
                    st.metric(
                        "📊 Change vs Average",
                        f"{change:+.1f}%",
                        delta_color="normal" if change > 0 else "inverse"
                    )
                
                st.info("💡 This forecast is based on historical monthly sales patterns using machine learning algorithms.")
            except Exception as e:
                st.error(f"⚠️ Prediction Error: {str(e)}")
    else:
        st.warning("⚠️ Model not loaded. Please ensure 'monthly_sales_model.pkl' exists.")

# ===============================
# 4. ADVANCED ANALYTICS
# ===============================
elif option == "🔍 Analytics":
    st.markdown("### 🔍 Deep Dive Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Product Analysis", "🏪 Category Insights", "⏰ Temporal Analysis"])
    
    with tab1:
        st.markdown("#### Top & Bottom Products")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🏆 Top 10 Best Sellers")
            top_10 = filtered_df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
            fig = go.Figure(data=[
                go.Bar(x=top_10.values, y=top_10.index, orientation='h',
                       marker=dict(color='#22C55E'))
            ])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#E0E0E0'), showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📉 Bottom 10 Products")
            bottom_10 = filtered_df.groupby("Product Name")["Sales"].sum().sort_values(ascending=True).head(10)
            fig = go.Figure(data=[
                go.Bar(x=bottom_10.values, y=bottom_10.index, orientation='h',
                       marker=dict(color='#EF4444'))
            ])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#E0E0E0'), showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Category Deep Dive")
        selected_category = st.selectbox("Choose Category:", filtered_df["Category"].unique())
        cat_data = filtered_df[filtered_df["Category"] == selected_category]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Category Total", f"${cat_data['Sales'].sum():,.2f}")
        with col2:
            st.metric("📦 Products Count", f"{cat_data['Product Name'].nunique()}")
        with col3:
            st.metric("📊 Avg Order", f"${cat_data['Sales'].mean():,.2f}")
        
        subcategory_sales = cat_data.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        fig = go.Figure(data=[
            go.Bar(x=subcategory_sales.index, y=subcategory_sales.values,
                   marker=dict(color=subcategory_sales.values, colorscale='Plasma'))
        ])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='#E0E0E0'), height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Temporal Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📅 Sales by Month")
            monthly = filtered_df.groupby(filtered_df["Order Date"].dt.month)["Sales"].sum()
            fig = go.Figure(data=[
                go.Bar(x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(monthly)],
                       y=monthly.values, marker=dict(color='#3B82F6'))
            ])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#E0E0E0'), height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 Sales by Day of Week")
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy["DayOfWeek"] = filtered_df_copy["Order Date"].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily = filtered_df_copy.groupby("DayOfWeek")["Sales"].sum().reindex(day_order, fill_value=0)
            fig = go.Figure(data=[
                go.Bar(x=daily.index, y=daily.values, marker=dict(color='#A78BFA'))
            ])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#E0E0E0'), height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("<p style='text-align: center; color: #9D4EDD; font-size: 12px;'>✨ Supermart Analytics Dashboard | Built with Streamlit & Cosmic Design</p>", unsafe_allow_html=True)
