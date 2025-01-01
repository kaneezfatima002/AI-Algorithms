import pandas as pd
import numpy as np

def calculate_entropy(data, target_col):
    values, counts = np.unique(data[target_col], return_counts=True)
    entropy = -sum((counts[i] / sum(counts)) * np.log2(counts[i] / sum(counts)) for i in range(len(values)))
    return entropy

def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    values, counts = np.unique(data[attribute], return_counts=True)

    weighted_entropy = sum(
        (counts[i] / sum(counts)) * calculate_entropy(data[data[attribute] == values[i]], target_col)
        for i in range(len(values))
    )
    return total_entropy - weighted_entropy


def build_tree(data, attributes, target_col):
    if len(np.unique(data[target_col])) == 1:
        return np.unique(data[target_col])[0]
    if len(attributes) == 0:
        return data[target_col].mode()[0]

    gains = {attr: calculate_information_gain(data, attr, target_col) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}

    for value in np.unique(data[best_attr]):
        subset = data[data[best_attr] == value]
        subtree = build_tree(subset, [attr for attr in attributes if attr != best_attr], target_col)
        tree[best_attr][value] = subtree

    return tree


def predict(tree, data_point, default=None):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = data_point.get(attr, None)
    if value not in tree[attr]:
        return default
    return predict(tree[attr][value], data_point, default)


data = pd.DataFrame({
    "Weather": ["Sunny", "Overcast", "Rainy", "Sunny", "Rainy"],
    "Temperature": ["Hot", "Hot", "Mild", "Cool", "Cool"],
    "Play": ["No", "Yes", "Yes", "Yes", "No"]
})

target_col = "Play"
attributes = list(data.columns[:-1])

decision_tree = build_tree(data, attributes, target_col)
print("Decision Tree:", decision_tree)

default_class = data[target_col].mode()[0]

test_data = pd.DataFrame({
    "Weather": ["Sunny", "Rainy", "Sunny"],
    "Temperature": ["Cool", "Hot", "Mild"]
})
for _, row in test_data.iterrows():
    prediction = predict(decision_tree, row, default=default_class)
    print(f"Prediction for {row.to_dict()}: {prediction}")
