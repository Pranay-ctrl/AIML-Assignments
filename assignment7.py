import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Create sample dataset
X = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
).reshape(-1, 1)

y = np.array([12, 18, 25, 31, 39, 44, 52, 58, 65, 71])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create the SVR model
model = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=100, epsilon=2)
)

# Train the model
model.fit(X_train, y_train)

# Predict test values
predictions = model.predict(X_test)

# Display results
print("Actual Values:", y_test)
print("Predicted Values:", predictions)

print("MAE:", mean_absolute_error(y_test, predictions))

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

print("R2 Score:", r2_score(y_test, predictions))