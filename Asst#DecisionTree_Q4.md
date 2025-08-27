# Question 4: Runtime Complexity Analysis

## Timing Plots

The following plots show the time taken for training (fit) and prediction (predict) as the number of samples (N) and features (M) are varied.

![Runtime Analysis]
<img width="1200" height="810" alt="q4_runtime_analysis" src="https://github.com/user-attachments/assets/2a28d797-ca14-4d25-8aca-3b3b2dcff59a" />

---

## Comparison with Theoretical Time Complexity

**Theoretical Complexity:**
-   **Training (Fit):** O(M * N * log(N))
-   **Prediction (Predict):** O(N * D), where D is the tree depth.

**Analysis of Experimental Results:**

1.  **Fit Time vs. N (Top-Left):** The plot shows a super-linear relationship. The time taken grows faster than the number of samples, which is consistent with the expected **N * log(N)** complexity.

2.  **Predict Time vs. N (Top-Right):** The plot shows a clear linear relationship. The time taken is directly proportional to the number of samples, which matches the expected **O(N)** complexity.

3.  **Fit Time vs. M (Bottom-Left):** The plot shows a linear relationship. The training time increases proportionally with the number of features, which is consistent with the **O(M)** component of the complexity.

4.  **Predict Time vs. M (Bottom-Right):** The plot is mostly flat with some noise. This correctly shows that prediction time is independent of the number of features, as the tree structure is already fixed. The noise is likely due to the very short measurement times.

**Conclusion:** The experimental results align well with the theoretical time complexities for decision tree creation and prediction.
