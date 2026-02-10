# Task 4 – Machine Learning & Visualisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# change path if needed
df = pd.read_csv("taxi.csv")

print("Dataset shape:", df.shape)
print(df.head())



# Remove missing values
df = df.dropna()

# Convert datetime to useful numerical features
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month

# Select useful features
features = [
    'passenger_count',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'hour',
    'day',
    'month'
]

target = 'trip_duration'

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))


# Scatter: Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.5)
plt.xlabel("Actual Duration")
plt.ylabel("Predicted Duration")
plt.title("Actual vs Predicted Trip Duration")
plt.show()


# Feature Importance (coefficients)
importance = pd.Series(model.coef_, index=features)

plt.figure(figsize=(6,4))
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.xlabel("Coefficient Value")
plt.show()


print("\nFeature Importance:")
print(importance.sort_values(ascending=False))
