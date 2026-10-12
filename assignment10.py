import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Artificial teaching examples
# 0 = Real, 1 = Fake
texts = [
    "City council publishes the approved annual budget",
    "Health department releases its vaccination schedule",
    "University announces examination timetable",
    "Officials publish verified rainfall measurements",
    "Secret miracle cure works instantly for everyone",
    "Shocking claim says the moon disappeared overnight",
    "Share now or your phone will be permanently blocked",
    "Unbelievable story claims all schools are closed forever"
]

labels = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1
])

# Demonstration image features:
# Average brightness, average red, average blue.
# These values are not extracted from actual images.
image_features = np.array([
    [0.45, 0.40, 0.42],
    [0.50, 0.46, 0.44],
    [0.48, 0.43, 0.41],
    [0.44, 0.39, 0.40],
    [0.80, 0.75, 0.70],
    [0.85, 0.78, 0.76],
    [0.75, 0.72, 0.68],
    [0.82, 0.79, 0.73]
])

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english"
)

text_features = vectorizer.fit_transform(texts)

# Convert image features into a sparse matrix
image_matrix = csr_matrix(image_features)

# Combine text and image features
X = hstack([
    text_features,
    image_matrix
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# Create and train the classifier
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predict test labels
predictions = model.predict(X_test)

# Display evaluation results
print(
    "Demonstration Accuracy:",
    accuracy_score(y_test, predictions)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Test a new example
new_text = [
    "Officials announce the verified public transport schedule"
]

new_text_features = vectorizer.transform(new_text)

# Example image features only
new_image_features = csr_matrix([
    [0.46, 0.41, 0.43]
])

# Combine new text and image features
new_data = hstack([
    new_text_features,
    new_image_features
])

# Predict the class
result = model.predict(new_data)[0]

if result == 1:
    print("Demo Prediction: FAKE")
else:
    print("Demo Prediction: REAL")