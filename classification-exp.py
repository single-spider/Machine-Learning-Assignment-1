import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

X = pd.DataFrame(X, columns=['feature1', 'feature2'])
y = pd.Series(y)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
plt.title("Generated Dataset")
plt.show()

# Q2 a)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTree(criterion='information_gain', max_depth=5)
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

print("--- Q2 a) Results ---")
print("Accuracy:", accuracy(y_hat, y_test))
for cls in y.unique():
    print(f"Class {cls} Precision:", precision(y_hat, y_test, cls))
    print(f"Class {cls} Recall:", recall(y_hat, y_test, cls))

# Q2 b)
print("\n--- Q2 b) 5-Fold Cross-Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
depths = range(1, 11)
best_depth = -1
best_accuracy = -1

for depth in depths:
    accuracies = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        tree = DecisionTree(max_depth=depth)
        tree.fit(X_train_fold, y_train_fold)
        y_hat_fold = tree.predict(X_val_fold)
        accuracies.append(accuracy(y_hat_fold, y_val_fold))
    
    avg_accuracy = np.mean(accuracies)
    print(f"Depth: {depth}, Average Accuracy: {avg_accuracy}")
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_depth = depth

print(f"\nOptimal Depth: {best_depth} with accuracy: {best_accuracy}")
