# Question 2: Classification Experiment

## Generated Dataset Visualization

The dataset was generated with two informative features and two classes. The scatter plot below shows the distribution of the data points.

<img width="640" height="480" alt="gen-dataset" src="https://github.com/user-attachments/assets/f49b955c-585f-4a21-9834-e1178707283e" />
[Generated Dataset]

---

## a) Performance on Test Set

The model was trained on 70% of the data and tested on the remaining 30%. The following metrics were achieved:
```
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
```
**Analysis:** The optimal depth for the tree is 2. Performance peaks at this depth and then begins to decrease, which indicates that deeper trees are overfitting to the training data

Results for Test Usage

[Generated Trees]
<img width="1536" height="759" alt="Figure_1" src="https://github.com/user-attachments/assets/5a82c964-596e-4bf0-b4ca-0678877a1998" />
<img width="1536" height="759" alt="Figure_2" src="https://github.com/user-attachments/assets/9bfbbf8b-2efb-42af-9e94-9aa5bd0a3d4d" />
<img width="1500" height="810" alt="Figure_3" src="https://github.com/user-attachments/assets/59c95b05-850d-4c80-8d61-af7b2068ddf4" />
<img width="1500" height="810" alt="Figure_4" src="https://github.com/user-attachments/assets/b79a08e9-471b-4580-b6b4-f7091b52bcef" />
<img width="1500" height="810" alt="Figure_5" src="https://github.com/user-attachments/assets/89c51c1d-4ed4-4f96-839e-4129f6b8f863" />
<img width="1500" height="810" alt="Figure_6" src="https://github.com/user-attachments/assets/0a71ae42-0c8a-438c-80cd-d93965a5fe61" />
<img width="1500" height="810" alt="Figure_7" src="https://github.com/user-attachments/assets/49d3266d-b9da-43c9-b74e-c9f398587c2b" />
<img width="1500" height="810" alt="Figure_8" src="https://github.com/user-attachments/assets/1d68f30b-40b2-4ebb-b27d-4e64f2ca3f6b" />
