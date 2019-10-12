## ETH Zurich Advanced Machine Learning Course Projects
### Project 1: Age Prediction from MRI Features
In this project we received a perturbeded MRI features data and was asked to predict the ages of the people using that data.
To solve this project we used the following pipeline:
1. Impute the data. Our given data was inserted with NaN values to simulate a situation where some features had uncomplete data. To overcome this problem, we used _missForest_ algorithm implemeted in R.
2. After imputing the lost data, because we knew that our data also had outliers, we used the _RobustScaler_ of sklearn package.
3. Our data also had false features: the number of features needed to actually predict age from MRI pictures is ~250 features, however, our data had 860. Therefore, we selected only the top 30% data explaining the targets.
4. After reducing the number of features, we used the _IsolationForest_ algorithm to detect 99 outlier samples in our data. 
5. Finally, after the pre-processing stage, we used the _GradientBoostingRegressor_ to fit a model out of the training set data.
6. To validate our pipline, we used a cross-validation procedure of 3 folds with R^2 score metric.
7. Use our model to predict the ages of the test data set.

