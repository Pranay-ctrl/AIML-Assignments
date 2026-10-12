from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create the classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict test labels
predictions = model.predict(X_test)

# Display results
print("Predicted Labels:")
print(predictions[:10])

print("Actual Labels:")
print(y_test[:10])

print("Accuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)