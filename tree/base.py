from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.features_ = X.columns.tolist()
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(y.unique())

        if (depth >= self.max_depth or n_labels == 1 or n_samples < 2):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        best_feature_idx, best_thresh = opt_split_attribute(X, y, self.criterion)
        
        if best_feature_idx is None:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        best_feature_name = X.columns[best_feature_idx]

        if pd.api.types.is_numeric_dtype(X[best_feature_name]):
            left_mask = X[best_feature_name] <= best_thresh
            right_mask = X[best_feature_name] > best_thresh
        else:
            left_mask = X[best_feature_name] == best_thresh
            right_mask = X[best_feature_name] != best_thresh

        left_X, left_y = X.loc[left_mask], y.loc[left_mask]
        right_X, right_y = X.loc[right_mask], y.loc[right_mask]

        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)
        return Node(best_feature_idx, best_thresh, left, right)

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
        
        feature_name = self.features_[node.feature]

        if pd.api.types.is_numeric_dtype(x[feature_name]):
            if x[feature_name] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if x[feature_name] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def _assign_positions(self, node, depth=0):
        """
        Assigns (x, y) coordinates to each node using a recursive layout algorithm.
        Returns a dictionary of {node: (x, y)} and the width of the subtree.
        """
        if node.is_leaf_node():
            return {node: (0, -depth)}, 1

        left_pos, left_width = self._assign_positions(node.left, depth + 1)
        right_pos, right_width = self._assign_positions(node.right, depth + 1)

        for n, (x, y) in right_pos.items():
            right_pos[n] = (x + left_width + 1, y)

        positions = {**left_pos, **right_pos}
        
        left_root_x = left_pos[node.left][0]
        right_root_x = right_pos[node.right][0]
        node_x = (left_root_x + right_root_x) / 2
        positions[node] = (node_x, -depth)

        total_width = left_width + right_width + 1
        return positions, total_width

    def plot(self, title="", metrics_text="") -> None:
        """Function to plot the tree using Matplotlib."""
        if self.root is None:
            print("Tree has not been fitted yet.")
            return

        positions, _ = self._assign_positions(self.root)
        
        if positions:
            root_x = positions[self.root][0]
            for node in positions:
                x, y = positions[node]
                positions[node] = (x - root_x, y)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('off')

        for node, (x, y) in positions.items():
            bbox_props = dict(boxstyle="round,pad=0.4", fc="lightblue", ec="b", lw=1)
            
            if node.is_leaf_node():
                node_text = f"Value = {node.value:.2f}" if isinstance(node.value, float) else f"Class = {node.value}"
            else:
                feature_name = self.features_[node.feature]
                threshold_str = f"{node.threshold:.2f}" if isinstance(node.threshold, (int, float)) else f"{node.threshold}"
                node_text = f"{feature_name}\n<= {threshold_str}"

            ax.text(x, y, node_text, ha="center", va="center", size=8, bbox=bbox_props)

            if not node.is_leaf_node():
                x_left, y_left = positions[node.left]
                x_right, y_right = positions[node.right]
                ax.plot([x, x_left], [y, y_left], "k-", lw=1)
                ax.plot([x, x_right], [y, y_right], "k-", lw=1)

        plt.title(title, size=16)
        fig.text(0.05, 0.05, metrics_text, ha="left", va="bottom", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="black", lw=1))
        plt.tight_layout()
        plt.show()
