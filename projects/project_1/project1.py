# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:01:27 2019

@author: oforster, slionar, yardas
@group : Mlcodebreakers
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, MissingIndicator


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Run a regression pipeline')
    parser.add_argument('--train', dest='train', required=True, help='Relative path of the training data.')
    parser.add_argument('--target', dest='target', required=True, help='Relative path of the target data.')
    parser.add_argument('--test', dest='test', required=True, help='Relative of the test data.')
    parser.add_argument('--predict', dest='predict', required=True, help='Relative of the prediction.')
    parser.add_argument('--n_folds', dest='n_folds', type=int, default=3,
                        help='Number of folds for the cross-validation. Default: 3')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print(args)
    return args


def main():
    options = parse_args()
    # read data from csv
    x_train_pd = pd.read_csv(options.train)
    x_train = x_train_pd.values[:, 1:]
    x_test_pd = pd.read_csv(options.test)
    x_test = x_test_pd.values[:, 1:]
    y_train_pd = pd.read_csv(options.target)
    y_train = y_train_pd.values[:, 1:]

    print('\nINSPECTION OF THE DATA')
    print('Size of X_train:\t', np.shape(x_train),
          '\nSize of X_test:\t', np.shape(x_test),
          '\nSize of y_train:\t', np.shape(y_train))

    print('\nFEATURE ENGINEERING')
    counter = 0
    for sample_id in range(x_train.shape[0]):
        if ~(np.isfinite(x_train[sample_id, :]).all()):
            counter += 1

    print("%d corrupt samples out of %d samples." %
          (counter, x_train.shape[0]))
    print("Junk percentage: %0.3f%%" % (x_train[~np.isfinite(x_train)].shape[0]
                                        / x_train.size * 100.))

    # Data NaNs imputation
    # imp = IterativeImputer(n_nearest_features=125)
    # imp.fit(x_train)
    # x_train = imp.transform(x_train)
    # Data normalization
    scaler = RobustScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    print('\nDATA NORMALIZATION REPORT')
    print('Mean of X_train:\t', x_train.mean(),
          '\nStd of X_train:\t', x_train.std())

    # Select important features
    print('\nFEATURES SELECTION')
    selector = SelectPercentile(percentile=28) \
        .fit(x_train, y_train.ravel())
    x_train = selector.transform(x_train)
    print("Number of features after selection: %d" % x_train.shape[1])

    # Detect outliers
    print('\nOUTLIER DETECTION')
    detector = IsolationForest(behaviour='new', max_samples=200,
                               contamination='auto')
    detector.fit(x_train)
    inliers_mask = detector.predict(x_train) == 1
    x_train = x_train[inliers_mask]
    print('Found %d outlier samples.' %
          (inliers_mask.shape[0] - np.sum(inliers_mask)))
    y_train = y_train[inliers_mask]

    print('\nAPPLY LEARNING METHOD')
    assert (np.isfinite(x_train).all() and np.isfinite(y_train).all())
    # params = {'n_estimators': np.linspace(100, 250, num=2, dtype=int),
    #           'max_depth': [2, 3, 4],
    #           'learning_rate': np.linspace(0.05, 0.15, num=3)}
    params = {'n_estimators': 350,
              'max_depth': 3,
              'learning_rate': 0.12,
              'loss': 'huber'}
    reg = GradientBoostingRegressor(**params). \
        fit(x_train, y_train.ravel())
    # best = GridSearchCV(reg, params, cv=5)
    # best_fit = best.fit(x_train, y_train.ravel())
    # reg = RandomForestRegressor(n_estimators=50). \
    #     fit(x_train, y_train.ravel())
    print("Training Coefficient of Determination (R^2): %0.4f" %
          reg.score(x_train, y_train.ravel()))
    # Cross validate
    scores = cross_validate(reg, x_train, y_train.ravel(),
                            scoring='r2',
                            return_train_score=True,
                            cv=options.n_folds)
    print("Test scores:\t", scores['test_score'],
          '\nTrain scores:\t', scores['train_score'])
    # x_test = imp.transform(x_test)
    assert (np.isfinite(x_test).all())
    x_test = scaler.transform(x_test)
    x_test = selector.transform(x_test)
    y_pred = reg.predict(x_test)

    # Write prediction into solution.csv
    id = [float(i) for i in range(0, len(y_pred))]
    df = pd.DataFrame({'id': id, 'y': y_pred})
    df.to_csv(options.predict, index=False)
    print('\nPrediction written to solution.csv')


if __name__ == '__main__':
    main()
