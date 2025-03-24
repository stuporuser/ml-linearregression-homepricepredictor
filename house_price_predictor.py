import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load sample dataset (or replace with real data)
data = {
    "SquareFeet": [750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400],
    "Price": [150000, 160000, 170000, 180000, 195000, 210000, 230000, 250000, 270000, 290000]
}
df = pd.DataFrame(data)

# Split into training and testing sets
X = df[["SquareFeet"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"m = {model.coef_[0]:.2f}")
print(f"b = {model.intercept_:.2f}")

# Visualize results
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Prices")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.legend()
plt.show()
