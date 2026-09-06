import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Thiranex Data Analytics Intership\Task3\train.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 3. Convert Order Date
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)


# 4. Create monthly sales
monthly_sales = df.resample('ME', on='Order Date')['Sales'].sum()

print("\nMonthly Sales:")
print(monthly_sales.head())


# 5. Plot historical sales
plt.figure(figsize=(12, 5))

plt.plot(monthly_sales)

plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.show()


# 6. Prepare data for Linear Regression
X = np.arange(len(monthly_sales)).reshape(-1, 1)
y = monthly_sales.values


# 7. Split data into training and testing
train_size = int(len(monthly_sales) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# 8. Train model
model = LinearRegression()

model.fit(X_train, y_train)


# 9. Predict test data
y_pred = model.predict(X_test)


# 10. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# 11. Predict next 6 months
future_X = np.arange(
    len(monthly_sales),
    len(monthly_sales) + 6
).reshape(-1, 1)

future_predictions = model.predict(future_X)


# 12. Create future dates
future_dates = pd.date_range(
    start=monthly_sales.index[-1] + pd.offsets.MonthEnd(1),
    periods=6,
    freq='ME'
)


# 13. Create forecast table
forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted Sales': future_predictions
})

print("\nFuture Sales Forecast:")
print(forecast)


# 14. Plot forecast
plt.figure(figsize=(12, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_predictions,
    label="Future Prediction"
)

plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()

plt.show()