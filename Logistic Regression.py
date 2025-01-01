import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        errors = predictions - y
        gradient = np.dot(X.T, errors) / m
        weights -= learning_rate * gradient

        if i % 100 == 0:
            loss = cross_entropy_loss(y, predictions)
            print(f"Iteration {i}, Loss: {loss}")
    return weights

def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return probabilities >= 0.5

def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    weights = np.zeros(X.shape[1])
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights

def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

X = np.array([
    [0.1, 1.1], [1.2, 0.9], [1.5, 1.6], [2.0, 1.8],
    [2.5, 2.1], [0.5, 1.5], [1.8, 2.3], [0.2, 0.7],
    [1.9, 1.4], [0.8, 0.6]
])
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

X = np.c_[np.ones(X.shape[0]), X]

plt.scatter(X[:, 1][y == 0], X[:, 2][y == 0], color='red', label='Class 0')
plt.scatter(X[:, 1][y == 1], X[:, 2][y == 1], color='blue', label='Class 1')
plt.xlabel('Feature 1 (X1)')
plt.ylabel('Feature 2 (X2)')
plt.legend()
plt.show()

learning_rate = 0.1
iterations = 1000
weights = logistic_regression(X, y, learning_rate, iterations)

y_pred = predict(X, weights)

accuracy = evaluate(y, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

x_values = np.linspace(-2, 2, 100)
y_values = -(weights[0] + weights[1] * x_values) / weights[2]
plt.scatter(X[:, 1][y == 0], X[:, 2][y == 0], color='red', label='Class 0')
plt.scatter(X[:, 1][y == 1], X[:, 2][y == 1], color='blue', label='Class 1')
plt.plot(x_values, y_values, color='green', label='Decision Boundary')
plt.xlabel('Feature 1 (X1)')
plt.ylabel('Feature 2 (X2)')
plt.legend()
plt.show()
