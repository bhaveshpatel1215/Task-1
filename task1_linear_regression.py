# Task 1 - House Price Prediction using Linear Regression
# Features: square footage, bedrooms and bathrooms

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the Kaggle House Prices dataset
# Put train.csv in the same folder as this file.
df = pd.read_csv("train.csv")

# 2. Select the required columns
# GrLivArea = above-ground living area (square feet)
# BedroomAbvGr = number of bedrooms above ground
# FullBath + 0.5*HalfBath = approximate number of bathrooms
df["Bathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]

X = df[["GrLivArea", "BedroomAbvGr", "Bathrooms"]]
y = df["SalePrice"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict house prices
y_pred = model.predict(X_test)

# 6. Check model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model")
print("-----------------------")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

print("\nModel Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# 7. Example prediction
example = pd.DataFrame({
    "GrLivArea": [1500],
    "BedroomAbvGr": [3],
    "Bathrooms": [2]
})
predicted_price = model.predict(example)[0]
print(f"\nPredicted price for 1500 sq ft, 3 bedrooms, 2 bathrooms: ${predicted_price:,.2f}")

# 8. Actual vs Predicted graph
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()
