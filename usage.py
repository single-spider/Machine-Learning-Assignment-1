
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Test case 1: Real Input and Real Output
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P), columns=[f'X{i}' for i in range(P)])
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=4)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    
    rmse_val = rmse(y_hat, y)
    mae_val = mae(y_hat, y)
    
    title = f"Real Input, Real Output (Criterion: {criteria})"
    metrics_str = f"RMSE: {rmse_val:.4f}\nMAE: {mae_val:.4f}"
    
    tree.plot(title=title, metrics_text=metrics_str)
    
    print(f"\n--- {title} ---")
    print("RMSE: ", rmse_val)
    print("MAE: ", mae_val)

# Test case 2: Real Input and Discrete Output
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P), columns=[f'X{i}' for i in range(P)])
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=4)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    
    acc = accuracy(y_hat, y)
    metrics_str = f"Accuracy: {acc:.4f}\n\n"
    for cls in sorted(y.unique()):
        prec = precision(y_hat, y, cls)
        rec = recall(y_hat, y, cls)
        metrics_str += f"Class {cls}:  Precision: {prec:.4f}  |  Recall: {rec:.4f}\n"
    
    title = f"Real Input, Discrete Output (Criterion: {criteria})"
    tree.plot(title=title, metrics_text=metrics_str)
    
    print(f"\n--- {title} ---")
    print("Accuracy: ", acc)
    for cls in y.unique():
        print(f"Class {cls}: Precision: {precision(y_hat, y, cls):.4f}, Recall: {recall(y_hat, y, cls):.4f}")

# Test case 3: Discrete Input and Discrete Output
N = 30
P = 5
X = pd.DataFrame({f'X{i}': pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=4)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    
    acc = accuracy(y_hat, y)
    metrics_str = f"Accuracy: {acc:.4f}\n\n"
    for cls in sorted(y.unique()):
        prec = precision(y_hat, y, cls)
        rec = recall(y_hat, y, cls)
        metrics_str += f"Class {cls}:  Precision: {prec:.4f}  |  Recall: {rec:.4f}\n"
    
    title = f"Discrete Input, Discrete Output (Criterion: {criteria})"
    tree.plot(title=title, metrics_text=metrics_str)
    
    print(f"\n--- {title} ---")
    print("Accuracy: ", acc)
    for cls in y.unique():
        print(f"Class {cls}: Precision: {precision(y_hat, y, cls):.4f}, Recall: {recall(y_hat, y, cls):.4f}")

# Test case 4: Discrete Input and Real Output
N = 30
P = 5
X = pd.DataFrame({f'X{i}': pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=4)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    
    rmse_val = rmse(y_hat, y)
    mae_val = mae(y_hat, y)
    
    title = f"Discrete Input, Real Output (Criterion: {criteria})"
    metrics_str = f"RMSE: {rmse_val:.4f}\nMAE: {mae_val:.4f}"
    
    tree.plot(title=title, metrics_text=metrics_str)
    
    print(f"\n--- {title} ---")
    print("RMSE: ", rmse_val)
    print("MAE: ", mae_val)
