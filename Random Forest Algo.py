# import math
# import random
# #TASK1
# data = [
#     {"Weather": "Sunny", "Temperature": "Hot", "Play?": "No"},
#     {"Weather": "Overcast", "Temperature": "Hot", "Play?": "Yes"},
#     {"Weather": "Rainy", "Temperature": "Mild", "Play?": "Yes"},
#     {"Weather": "Sunny", "Temperature": "Mild", "Play?": "No"},
#     {"Weather": "Overcast", "Temperature": "Mild", "Play?": "Yes"},
#     {"Weather": "Rainy", "Temperature": "Hot", "Play?": "No"}
# ]
# def count_occurrences(values):
#     counts = {}
#     for value in values:
#         if value in counts:
#             counts[value] += 1
#         else:
#             counts[value] = 1
#     return counts
#
# def calculate_entropy(data, target_col):
#     values = [row[target_col] for row in data]
#     counts = count_occurrences(values)
#     total = len(values)
#     entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
#     return entropy
#
#
# def calculate_information_gain(data, attribute, target_col):
#     total_entropy = calculate_entropy(data, target_col)
#     values = set(row[attribute] for row in data)
#     weighted_entropy = 0
#     for value in values:
#         subset = [row for row in data if row[attribute] == value]
#         subset_entropy = calculate_entropy(subset, target_col)
#         weighted_entropy += (len(subset) / len(data)) * subset_entropy
#     return total_entropy - weighted_entropy
#
#
# def build_tree(data, attributes, target_col, depth=0, max_depth=3):
#     values = [row[target_col] for row in data]
#     counts = count_occurrences(values)
#     most_common = max(counts, key=counts.get)
#
#     if len(counts) == 1 or depth == max_depth or not attributes:
#         return most_common
#
#     best_attribute = max(attributes, key=lambda attr: calculate_information_gain(data, attr, target_col))
#     tree = {best_attribute: {}}
#
#     for value in set(row[best_attribute] for row in data):
#         subset = [row for row in data if row[best_attribute] == value]
#         if not subset:
#             tree[best_attribute][value] = most_common
#         else:
#             tree[best_attribute][value] = build_tree(subset, [attr for attr in attributes if attr != best_attribute], target_col, depth + 1, max_depth)
#     return tree
#
#
# def predict(tree, data_point):
#     if not isinstance(tree, dict):
#         return tree
#     attribute, branches = list(tree.items())[0]
#     value = data_point.get(attribute)
#     return predict(branches.get(value, None), data_point) if value in branches else None
#
#
# def build_random_forest(data, attributes, target_col, n_trees=2):
#     forest = []
#     for _ in range(n_trees):
#         sample = random.choices(data, k=len(data))
#         forest.append(build_tree(sample, attributes, target_col))
#     return forest
#
#
# def random_forest_predict(forest, data_point):
#     predictions = [predict(tree, data_point) for tree in forest]
#     prediction_counts = count_occurrences(predictions)
#     return max(prediction_counts, key=prediction_counts.get)
#
#
# attributes = ["Weather", "Temperature"]
# target_col = "Play?"
#
#
# forest = build_random_forest(data, attributes, target_col)
#
#
# test_point = {"Weather": "Sunny", "Temperature": "Hot"}
# prediction = random_forest_predict(forest, test_point)
#
# print("Random Forest Algorithm:")
# print("Random Forest Prediction:", prediction)
#
#
#
# ###########################################################
#
# X = [1, 2, 3, 4, 5]
# Y = [2, 4, 5, 7, 8]
#
#
# def calculate_mean(values):
#     return sum(values) / len(values)
#
# def calculate_slope(X, Y, mean_X, mean_Y):
#     numerator = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(len(X)))
#     denominator = sum((X[i] - mean_X) ** 2 for i in range(len(X)))
#     return numerator / denominator
#
# def calculate_intercept(mean_X, mean_Y, slope):
#     return mean_Y - (slope * mean_X)
#
# def predict(X, theta_0, theta_1):
#     return [theta_0 + theta_1 * x for x in X]
#
#
# def calculate_mse(Y, Y_pred):
#     return sum((Y[i] - Y_pred[i]) ** 2 for i in range(len(Y))) / len(Y)
#
#
# def fit_linear_regression(X, Y):
#     mean_X = calculate_mean(X)
#     mean_Y = calculate_mean(Y)
#     slope = calculate_slope(X, Y, mean_X, mean_Y)
#     intercept = calculate_intercept(mean_X, mean_Y, slope)
#
#     return intercept, slope
#
# def main():
#     intercept, slope = fit_linear_regression(X, Y)
#     Y_pred = predict(X, intercept, slope)
#     mse = calculate_mse(Y, Y_pred)
#     print("Intercept (theta_0):", intercept)
#     print("Slope (theta_1):", slope)
#     print("Predictions (Y^):", Y_pred)
#     print("Mean Squared Error (MSE):", mse)
#
#
# print()
# print("Linear Regression:")
# main()

import pandas as pd
import math
import random

# Load the dataset
df = pd.read_csv('E:/AI lab/Employee.csv')

# Convert categorical columns into string type
df['Education'] = df['Education'].astype(str)
df['City'] = df['City'].astype(str)
df['Gender'] = df['Gender'].astype(str)
df['EverBenched'] = df['EverBenched'].astype(str)

# Prepare the dataset
data = df.to_dict(orient='records')
target_col = 'LeaveOrNot'  # Target column for prediction
attributes = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']  # Feature columns


# Function to count occurrences
def count_occurrences(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts

# Calculate entropy
def calculate_entropy(data, target_col):
    values = [row[target_col] for row in data]
    counts = count_occurrences(values)
    total = len(values)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy

# Calculate information gain
def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    values = set(row[attribute] for row in data)
    weighted_entropy = 0
    for value in values:
        subset = [row for row in data if row[attribute] == value]
        subset_entropy = calculate_entropy(subset, target_col)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    return total_entropy - weighted_entropy

# Build decision tree
def build_tree(data, attributes, target_col, depth=0, max_depth=3):
    values = [row[target_col] for row in data]
    counts = count_occurrences(values)
    most_common = max(counts, key=counts.get)

    if len(counts) == 1 or depth == max_depth or not attributes:
        return most_common

    best_attribute = max(attributes, key=lambda attr: calculate_information_gain(data, attr, target_col))
    tree = {best_attribute: {}}

    for value in set(row[best_attribute] for row in data):
        subset = [row for row in data if row[best_attribute] == value]
        if not subset:
            tree[best_attribute][value] = most_common
        else:
            tree[best_attribute][value] = build_tree(subset, [attr for attr in attributes if attr != best_attribute], target_col, depth + 1, max_depth)
    return tree

# Predict using tree
def predict(tree, data_point):
    if not isinstance(tree, dict):
        return tree
    attribute, branches = list(tree.items())[0]
    value = data_point.get(attribute)
    return predict(branches.get(value, None), data_point) if value in branches else None

# Build random forest
def build_random_forest(data, attributes, target_col, n_trees=5):
    forest = []
    for _ in range(n_trees):
        sample = random.choices(data, k=len(data))
        forest.append(build_tree(sample, attributes, target_col))
    return forest

# Predict using random forest
def random_forest_predict(forest, data_point):
    predictions = [predict(tree, data_point) for tree in forest]
    prediction_counts = count_occurrences(predictions)
    return max(prediction_counts, key=prediction_counts.get)


# Build Random Forest
forest = build_random_forest(data, attributes, target_col)

# Test prediction for a new employee
test_point = {
    'Education': 'Bachelors',
    'JoiningYear': 2016,
    'City': 'Bangalore',
    'PaymentTier': 3,
    'Age': 34,
    'Gender': 'Male',
    'EverBenched': 'No',
    'ExperienceInCurrentDomain': 4
}
prediction = random_forest_predict(forest, test_point)

print("Random Forest Prediction (LeaveOrNot):", prediction)
