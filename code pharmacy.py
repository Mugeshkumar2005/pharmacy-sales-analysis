import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("pharmacy_sales_5000_sorted (2).csv")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.head())

df = df.drop_duplicates()

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

df["Quantity Sold"] = pd.to_numeric(
    df["Quantity Sold"],
    errors="coerce"
)

df["Unit Price"] = pd.to_numeric(
    df["Unit Price"],
    errors="coerce"
)

df["Total Sales Amount"] = pd.to_numeric(
    df["Total Sales Amount"],
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

df["Quantity Sold"] = df["Quantity Sold"].fillna(
    df["Quantity Sold"].median()
)

df["Unit Price"] = df["Unit Price"].fillna(
    df["Unit Price"].median()
)

df["Customer Age"] = df["Customer Age"].fillna(
    df["Customer Age"].median()
)

df["Discount Applied"] = df["Discount Applied"].fillna(0)

df["Total Sales Amount"] = df["Total Sales Amount"].fillna(
    df["Total Sales Amount"].median()
)

df = df.dropna(subset=["Date"])

df.to_csv(
    "pharmacy_sales_cleaned.csv",
    index=False
)

print(df.isnull().sum())

print(df.describe())

plt.figure(figsize=(8, 5))
sns.histplot(
    df["Total Sales Amount"],
    kde=True
)
plt.title("Distribution of Total Sales")
plt.xlabel("Total Sales Amount")
plt.ylabel("Number of Transactions")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Quantity Sold",
    y="Total Sales Amount"
)
plt.title("Quantity Sold vs Total Sales")
plt.xlabel("Quantity Sold")
plt.ylabel("Total Sales Amount")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Unit Price",
    y="Total Sales Amount"
)
plt.title("Unit Price vs Total Sales")
plt.xlabel("Unit Price")
plt.ylabel("Total Sales Amount")
plt.show()

product_sales = (
    df.groupby("Product Type")["Total Sales Amount"]
    .sum()
    .sort_values(ascending=False)
)

print(product_sales)

plt.figure(figsize=(10, 5))
product_sales.plot(kind="bar")
plt.title("Sales by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

payment_sales = (
    df.groupby("Payment Method")["Total Sales Amount"]
    .sum()
    .sort_values(ascending=False)
)

print(payment_sales)

plt.figure(figsize=(8, 5))
payment_sales.plot(kind="bar")
plt.title("Sales by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

gender_sales = (
    df.groupby("Customer Gender")["Total Sales Amount"]
    .sum()
)

print(gender_sales)

plt.figure(figsize=(7, 5))
gender_sales.plot(kind="bar")
plt.title("Sales by Customer Gender")
plt.xlabel("Customer Gender")
plt.ylabel("Total Sales Amount")
plt.tight_layout()
plt.show()

df["Month"] = df["Date"].dt.month

monthly_sales = (
    df.groupby("Month")["Total Sales Amount"]
    .sum()
)

print(monthly_sales)

plt.figure(figsize=(9, 5))
monthly_sales.plot(
    kind="line",
    marker="o"
)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales Amount")
plt.grid()
plt.show()

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

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

quantity = 5
price = 50
discount = 10
age = 30

new_data = pd.DataFrame({
    "Quantity Sold": [quantity],
    "Unit Price": [price],
    "Discount Applied": [discount],
    "Customer Age": [age]
})

prediction = model.predict(new_data)

print("Quantity Sold:", quantity)
print("Unit Price:", price)
print("Discount:", discount, "%")
print("Customer Age:", age)

print(
    "Predicted Sales Amount:",
    round(prediction[0], 2)
)