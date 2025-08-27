import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree

def generate_data(N, M, output_type='discrete'):
    X = pd.DataFrame(np.random.randint(0, 2, size=(N, M)))
    if output_type == 'discrete':
        y = pd.Series(np.random.randint(0, 2, size=N))
    else:
        y = pd.Series(np.random.rand(N))
    return X, y

# Varying N
N_values = [10, 50, 100, 200, 500]
M = 5
fit_times_N = []
predict_times_N = []

for N in N_values:
    X, y = generate_data(N, M)
    tree = DecisionTree(max_depth=5)
    
    start = time.time()
    tree.fit(X, y)
    fit_times_N.append(time.time() - start)
    
    start = time.time()
    tree.predict(X)
    predict_times_N.append(time.time() - start)

# Varying M
M_values = [2, 5, 10, 15, 20]
N = 100
fit_times_M = []
predict_times_M = []

for M in M_values:
    X, y = generate_data(N, M)
    tree = DecisionTree(max_depth=5)
    
    start = time.time()
    tree.fit(X, y)
    fit_times_M.append(time.time() - start)
    
    start = time.time()
    tree.predict(X)
    predict_times_M.append(time.time() - start)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(N_values, fit_times_N, marker='o')
axs[0, 0].set_title('Fit Time vs. N (M=5)')
axs[0, 0].set_xlabel('Number of Samples (N)')
axs[0, 0].set_ylabel('Time (s)')

axs[0, 1].plot(N_values, predict_times_N, marker='o')
axs[0, 1].set_title('Predict Time vs. N (M=5)')
axs[0, 1].set_xlabel('Number of Samples (N)')
axs[0, 1].set_ylabel('Time (s)')

axs[1, 0].plot(M_values, fit_times_M, marker='o')
axs[1, 0].set_title('Fit Time vs. M (N=100)')
axs[1, 0].set_xlabel('Number of Features (M)')
axs[1, 0].set_ylabel('Time (s)')

axs[1, 1].plot(M_values, predict_times_M, marker='o')
axs[1, 1].set_title('Predict Time vs. M (N=100)')
axs[1, 1].set_xlabel('Number of Features (M)')
axs[1, 1].set_ylabel('Time (s)')

plt.tight_layout()
plt.show()
