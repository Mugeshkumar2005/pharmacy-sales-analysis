import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Pharmacy Sales Prediction",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Pharmacy Sales Prediction")

st.write(
    "This application uses Machine Learning to predict "
    "the expected sales amount for a pharmacy transaction."
)

df = pd.read_csv("pharmacy_sales_5000_sorted (2).csv")

df = df.drop_duplicates()

df["Quantity Sold"] = pd.to_numeric(
    df["Quantity Sold"],
    errors="coerce"
)

df["Unit Price"] = pd.to_numeric(
    df["Unit Price"],
    errors="coerce"
)

df["Customer Age"] = pd.to_numeric(
    df["Customer Age"],
    errors="coerce"
)

df["Discount Applied"] = (
    df["Discount Applied"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["Discount Applied"] = pd.to_numeric(
    df["Discount Applied"],
    errors="coerce"
)

df["Total Sales Amount"] = pd.to_numeric(
    df["Total Sales Amount"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Quantity Sold",
        "Unit Price",
        "Discount Applied",
        "Customer Age",
        "Total Sales Amount"
    ]
)

X = df[
    [
        "Quantity Sold",
        "Unit Price",
        "Discount Applied",
        "Customer Age"
    ]
]

y = df["Total Sales Amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

st.subheader("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total Records",
        len(df)
    )

with col2:
    st.metric(
        "Number of Features",
        4
    )

st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    quantity = st.number_input(
        "Quantity Sold",
        min_value=1,
        value=5
    )

    price = st.number_input(
        "Unit Price",
        min_value=0.0,
        value=50.0
    )

with col2:
    discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    age = st.number_input(
        "Customer Age",
        min_value=1,
        max_value=100,
        value=30
    )

if st.button("Predict Sales", type="primary"):

    new_data = pd.DataFrame({
        "Quantity Sold": [quantity],
        "Unit Price": [price],
        "Discount Applied": [discount],
        "Customer Age": [age]
    })

    prediction = model.predict(new_data)

    st.success(
        f"Predicted Sales Amount: ₹{prediction[0]:,.2f}"
    )

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )

st.subheader("Sales by Product Type")

product_sales = (
    df.groupby("Product Type")["Total Sales Amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(product_sales)