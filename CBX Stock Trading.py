import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data (replace with your file path)
file_path = "CBX Stock Trading.csv"
data = pd.read_csv(file_path)

# Convert 'date' to datetime and ordinal
data['date'] = pd.to_datetime(data['date'])
data['date_ordinal'] = data['date'].apply(lambda x: x.toordinal())

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Features and target selection
X = data[['date_ordinal']]  # Example: using only 'date_ordinal' as the feature
y = data['last_value']  # Target variable

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the fit
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Values')
plt.plot(X_test, y_pred, color='red', linestyle='dashed', label='Regression Line')
plt.xlabel('Date Ordinal')
plt.ylabel('Last Value')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
