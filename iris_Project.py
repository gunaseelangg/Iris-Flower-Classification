# Iris Flower Classification
# Internship Task 1
# Written by me

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
iris = load_iris()

# Step 2: Separate input and output
data = iris.data
labels = iris.target

# Step 3: Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.25, random_state=5
)

# Step 4: Create and train model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

# Step 5: Predict results
predicted_labels = knn_model.predict(x_test)

# Step 6: Check accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print("Model Accuracy:", accuracy)
# Step 5: Predict
y_pred = knn_model.predict(x_test)

# Step 6: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)