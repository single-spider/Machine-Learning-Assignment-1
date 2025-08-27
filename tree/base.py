from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from tree.utils import *

np.random.seed(42)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(y.unique())

        if (depth >= self.max_depth or n_labels == 1 or n_samples < 2):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        best_feature, best_thresh = opt_split_attribute(X, y, self.criterion)
        
        if best_feature is None:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        if pd.api.types.is_numeric_dtype(X[best_feature]):
            left_mask = X[best_feature] <= best_thresh
            right_mask = X[best_feature] > best_thresh
        else:
            left_mask = X[best_feature] == best_thresh
            right_mask = X[best_feature] != best_thresh

        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _leaf_value(self, y):
        if check_if_real(y):
            return y.mean()
        else:
            return y.mode().iloc[0]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._traverse_tree, axis=1, args=(self.root,))

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if pd.api.types.is_numeric_dtype(x[node.feature]):
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def plot(self) -> None:
        self._plot_tree(self.root)

    def _plot_tree(self, node, indent=""):
        if not node:
            return

        if node.is_leaf_node():
            print(indent, "->", node.value)
            return

        print(indent, f"?({node.feature} <= {node.threshold})")
        print(indent, "Y:")
        self._plot_tree(node.left, indent + "  ")
        print(indent, "N:")
        self._plot_tree(node.right, indent + "  ")
