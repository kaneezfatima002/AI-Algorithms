# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Perceptron implementation
# def perceptron(X, y, learning_rate=0.1, epochs=100):
#     """
#     Train a Perceptron model.
#     :param X: Input features (N x 2)
#     :param y: Target labels (N x 1)
#     :param learning_rate: Learning rate for weight updates
#     :param epochs: Number of iterations
#     :return: Trained weights and bias
#     """
#     # Initialize weights and bias
#     weights = np.zeros(X.shape[1])  # Two weights for two features
#     bias = 0
#
#     for _ in range(epochs):
#         for i in range(len(X)):
#             # Predict using the step function
#             prediction = 1 if np.dot(X[i], weights) + bias > 0 else 0
#             # Update weights and bias if prediction is wrong
#             error = y[i] - prediction
#             weights += learning_rate * error * X[i]
#             bias += learning_rate * error
#
#     return weights, bias
#
#
# # Visualize Perceptron Decision Boundary
# def plot_perceptron_boundary(X, y, weights, bias):
#     """
#     Plot decision boundary for the perceptron.
#     """
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
#     Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z > 0, alpha=0.7, cmap=plt.cm.Paired)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.Paired)
#     plt.title("Perceptron Decision Boundary")
#     plt.show()
#
#
# # Dataset for testing Perceptron
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # AND gate-like dataset
# y = np.array([0, 0, 0, 1])
#
# weights, bias = perceptron(X, y)
# plot_perceptron_boundary(X, y, weights, bias)
# from sklearn.neural_network import MLPClassifier
#
#
# # Train a Neural Network for XOR
# def train_xor_nn(X, y):
#     """
#     Train a simple neural network for XOR problem.
#     """
#     model = MLPClassifier(
#         hidden_layer_sizes=(2,),
#         max_iter=1000,
#         activation="logistic",
#         solver="adam",
#         random_state=42,
#     )
#     model.fit(X, y)
#     return model
#
#
# # Visualize XOR Decision Boundary
# def visualize_xor_boundary(model, X, y):
#     """
#     Plot decision boundary for the XOR problem.
#     """
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, alpha=0.7, cmap=plt.cm.Paired)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.Paired)
#     plt.title("XOR Neural Network Decision Boundary")
#     plt.show()
#
#
# # XOR Dataset
# X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_xor = np.array([0, 1, 1, 0])
#
# # Train and visualize
# xor_model = train_xor_nn(X_xor, y_xor)
# visualize_xor_boundary(xor_model, X_xor, y_xor)




import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a Neural Network with 2 Hidden Layers
nn_model = MLPClassifier(hidden_layer_sizes=(10, 8),  # Two hidden layers with 10 and 8 neurons
                          max_iter=1000,
                          activation='relu',
                          solver='adam',
                          random_state=42)

# Train the model
nn_model.fit(X_train, y_train)

# Predict on test data
y_pred = nn_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Neural Network: {accuracy:.2f}")

# Optional: Model Summary
print("\nModel Structure:")
print(f"Hidden Layers: {nn_model.hidden_layer_sizes}")
print(f"Number of Iterations: {nn_model.n_iter_}")
print(f"Loss: {nn_model.loss_:.4f}")

