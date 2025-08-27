# Question 2: Classification Experiment

## Generated Dataset Visualization

The dataset was generated with two informative features and two classes. The scatter plot below shows the distribution of the data points.

![Generated Dataset](q2_dataset.png)

---

## a) Performance on Test Set

The model was trained on 70% of the data and tested on the remaining 30%. The following metrics were achieved:

--- Q2 a) Results ---
Accuracy: 0.8333333333333334
Class 1 Precision: 0.8571428571428571
Class 1 Recall: 0.8
Class 0 Precision: 0.8125
Class 0 Recall: 0.8666666666666667

--- Q2 b) 5-Fold Cross-Validation ---
Depth: 1, Average Accuracy: 0.9
Depth: 2, Average Accuracy: 0.9200000000000002
Depth: 3, Average Accuracy: 0.9000000000000001
Depth: 4, Average Accuracy: 0.9000000000000001
Depth: 5, Average Accuracy: 0.89
Depth: 6, Average Accuracy: 0.8799999999999999
Depth: 7, Average Accuracy: 0.8699999999999999
Depth: 8, Average Accuracy: 0.8799999999999999
Depth: 9, Average Accuracy: 0.8699999999999999
Depth: 10, Average Accuracy: 0.8699999999999999

Optimal Depth: 2 with accuracy: 0.9200000000000002

**Analysis:** The optimal depth for the tree is 2. Performance peaks at this depth and then begins to decrease, which indicates that deeper trees are overfitting to the training data
