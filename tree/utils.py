import pandas as pd
import numpy as np

def check_if_real(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return y.dtype == 'float'

def entropy(y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    if y.size == 0:
        return 0
    counts = y.value_counts()
    probabilities = counts / y.size
    return -np.sum(probabilities * np.log2(probabilities))

def gini_index(y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    if y.size == 0:
        return 0
    counts = y.value_counts()
    probabilities = counts / y.size
    return 1 - np.sum(probabilities**2)

def information_gain(y: pd.Series, left_y: pd.Series, right_y: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain
    """
    if check_if_real(y):
        # Use MSE for real-valued outputs
        parent_mse = np.mean((y - y.mean())**2)
        left_mse = np.mean((left_y - left_y.mean())**2) if not left_y.empty else 0
        right_mse = np.mean((right_y - right_y.mean())**2) if not right_y.empty else 0
        
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        weighted_mse = (n_left / n) * left_mse + (n_right / n) * right_mse
        return parent_mse - weighted_mse
    else:
        # Use entropy or gini for discrete outputs
        if criterion == 'information_gain':
            impurity_func = entropy
        else:
            impurity_func = gini_index
            
        parent_impurity = impurity_func(y)
        left_impurity = impurity_func(left_y)
        right_impurity = impurity_func(right_y)
        
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        return parent_impurity - weighted_impurity

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion):
    """
    Function to find the optimal attribute to split about.
    """
    best_gain = -1
    best_feature = None
    best_value = None

    for feature in X.columns:
        values = X[feature].unique()
        for value in values:
            if pd.api.types.is_numeric_dtype(X[feature]):
                left_mask = X[feature] <= value
                right_mask = X[feature] > value
            else:
                left_mask = X[feature] == value
                right_mask = X[feature] != value
            
            left_y, right_y = y[left_mask], y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gain = information_gain(y, left_y, right_y, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value
                
    return best_feature, best_value
