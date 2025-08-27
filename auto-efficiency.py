import pandas as pd
import numpy as np
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

data = data.drop("car name", axis=1)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data = data.dropna()

X = data.drop("mpg", axis=1)
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# My Decision Tree
my_tree = DecisionTree(max_depth=5)
my_tree.fit(X_train, y_train)
my_y_hat = my_tree.predict(X_test)
print("--- My Decision Tree Performance ---")
print("RMSE:", rmse(my_y_hat, y_test))
print("MAE:", mae(my_y_hat, y_test))

# Scikit-learn Decision Tree
sklearn_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_y_hat = pd.Series(sklearn_tree.predict(X_test))
print("\n--- Scikit-learn Decision Tree Performance ---")
print("RMSE:", rmse(sklearn_y_hat, y_test))
print("MAE:", mae(sklearn_y_hat, y_test))
