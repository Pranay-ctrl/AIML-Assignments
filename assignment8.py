from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
data = load_diabetes()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create the models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "Lasso Regression": make_pipeline(
        StandardScaler(),
        Lasso(alpha=0.1, max_iter=10000)
    )
}

# Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    score = r2_score(y_test, predictions)

    print("\nModel:", name)
    print("Test RMSE:", rmse)
    print("Test R2 Score:", score)