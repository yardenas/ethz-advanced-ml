# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:01:27 2019

@author: oforster, slionar, yardas
@group : Mlcodebreakers
"""

import time
import math
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    parser.add_argument('--n_folds', dest='n_folds', type=int, default=5,
                        help='Number of folds for the cross-validation. Default: 5')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print(args)
    return args


# Prints the time since the last checkpoint (stored in checkpoint_time)
def print_time_since_checkpoint(checkpoint_time):
    
    if time.time()-checkpoint_time < 60:
        print('>>> ', round(time.time()-checkpoint_time, 3), 's')
    else:
        print('>>> ', math.floor((time.time()-checkpoint_time) / 60),
              'min', round(math.fmod(time.time()-checkpoint_time, 60)), 's')


def main():
    start_time = time.time()
    checkpoint_time = start_time

    options = parse_args()
    
    # read data from csv
    x_train_pd = pd.read_csv(options.train)
    x_train = x_train_pd.values[:, 1:]
    x_test_pd = pd.read_csv(options.test)
    x_test = x_test_pd.values[:, 1:]
    y_train_pd = pd.read_csv(options.target)
    y_train = y_train_pd.values[:, 1:]

    print('\nINSPECTION OF THE DATA')
    print('Size of X_train: ', np.shape(x_train))
    print('Size of X_test: ', np.shape(x_test))
    print('Size of y_train: ', np.shape(y_train))
    print_time_since_checkpoint(checkpoint_time)
    checkpoint_time = time.time()

    print('\nFEATURE ENGINEERING')
    counter = 0
    for sample_id in range(x_train.shape[0]):
        if ~(np.isfinite(x_train[sample_id, :]).all()):
            counter += 1

    print("%d corrupt samples out of %d samples" %
          (counter, x_train.shape[0]))
    print("Junk percentage: %0.3f%%" % (x_train[~np.isfinite(x_train)].shape[0]
          / x_train.size * 100.))

    # Data NaNs imputation
    imp = IterativeImputer(n_nearest_features=50)
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    # Data normalization
    scaler = RobustScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    print('\nDATA NORMALIZATION REPORT')
    print('Mean of X_train:', x_train.mean())
    print('Std of X_train:', x_train.std())

    print_time_since_checkpoint(checkpoint_time)
    print('\nAPPLY LEARNING METHOD')
    assert(np.isfinite(x_train).all())
    # params = {'n_estimators': 200, 'max_depth': 3,
    #           'learning_rate': 0.10, 'loss': 'huber'}
    # # reg = GradientBoostingRegressor(**params).fit(x_train, y_train.ravel())
    reg = RandomForestRegressor(n_estimators=50).\
        fit(x_train, y_train.ravel())
    print("Training Coefficient of Determination (R^2): %0.4f" %
          reg.score(x_train, y_train.ravel()))
    estimator = make_pipeline(
        make_union(imp, MissingIndicator()),
        reg)
    scores = cross_validate(estimator, x_train, y_train.ravel(),
                            scoring='r2',
                            cv=2)
    print("Test scores:\t", scores['test_score'],
          '\nTrain scores:\t', scores['train_score'])
    x_test = imp.transform(x_test)
    assert(np.isfinite(x_test).all())
    x_test = scaler.transform(x_test)
    y_pred = reg.predict(x_test)
    
    # Write prediction into solution.csv
    id = [float(i) for i in range(0, len(y_pred))]
    df = pd.DataFrame({'id': id, 'y': y_pred})
    df.to_csv(options.predict, index=False)
    print('Prediction written to solution.csv')
    print_time_since_checkpoint(start_time)
    
    
if __name__ == '__main__':
    main()
