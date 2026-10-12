import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Input data: study hours
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Output data: marks
y = np.array([35, 40, 50, 55, 65, 72])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict values
predictions = model.predict(X)

# Display model parameters
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Predict marks for 7 study hours
new_prediction = model.predict([[7]])
print("Predicted Marks for 7 Hours:", new_prediction[0])

# Evaluate the model
print("MAE:", mean_absolute_error(y, predictions))

rmse = np.sqrt(mean_squared_error(y, predictions))
print("RMSE:", rmse)

print("R2 Score:", r2_score(y, predictions))

# Plot actual values and regression line
plt.scatter(X, y, label="Actual Marks")
plt.plot(X, predictions, label="Regression Line")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)

plt.show()