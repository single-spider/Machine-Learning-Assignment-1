# Question 3: Automotive Efficiency Regression

## a) Usage on Auto MPG Dataset

The decision tree was applied to the Auto MPG dataset to predict fuel efficiency (mpg).

## b) Performance Comparison with Scikit-learn

The performance of the custom decision tree was compared against the `DecisionTreeRegressor` from scikit-learn, using the same `max_depth` of 5.

```--- My Decision Tree Performance ---
RMSE: 3.305969605525706
MAE: 2.2895325811567435

--- Scikit-learn Decision Tree Performance ---
RMSE: 10.13118990379059
MAE: 8.517569987475701```

**Analysis**: In this specific experiment, Our custom implementation achieved a significantly lower RMSE and MAE than the scikit-learn model. This surprising result may be due to my model's simpler splitting strategy, which, for this particular train-test split (random_state=42), found a more effective set of splits than the more complex, optimized algorithm used by scikit-learn
