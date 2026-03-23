# ==============================
# House Price Prediction Project
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 2. Load Dataset
df = pd.read_csv("house.csv")  # change name if needed

# 3. Data Preprocessing

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Handle categorical (e.g., location)
df = pd.get_dummies(df, drop_first=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# 4. Basic EDA

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 5. Feature & Target Split
# Assume target column is 'price'
X = df.drop("price", axis=1)
y = df["price"]

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 8. Model Training
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==============================
# 9. Evaluation
# ==============================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Model Evaluation ---")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# ==============================
# 10. Visualization
# ==============================

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# ==============================
# 11. Feature Importance
# ==============================

importance = pd.Series(model.coef_, index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# ==============================
# 12. Conclusion
# ==============================

print("\nProject Completed Successfully!")
print("Model predicts house prices using features like size, bedrooms, and location.")
print("Evaluation done using MAE and RMSE.")