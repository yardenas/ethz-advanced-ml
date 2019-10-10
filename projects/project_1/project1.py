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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import  SimpleImputer


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Run a regression pipeline')
    parser.add_argument('--train', dest='train', required=True, help='Relative path of the training data.')
    parser.add_argument('--target', dest='target', required=True, help='Relative path of the target data.')
    parser.add_argument('--test', dest='test', required=True, help='Relative of the test data.')
    parser.add_argument('--predict', dest='predict', required=True, help='Relative of the prediction.')
    parser.add_argument('--pca', dest='use_pca', action='store_true', help='Use PCA. Default: False')
    parser.add_argument('--pca_belief_ratio', dest='pca_belief_ratio', type=float, default=0.85,
                        help='Amount of variance that should be explained')
    parser.add_argument('--regression_normalize', dest='normalize', action='store_true',
                        help='Let RidgeCV normalize the training data. Default: False')
    parser.add_argument('--n_folds', dest='n_folds', type=int, default=5,
                        help='Number of folds for the cross-validation. Default: 5')
    if len(sys.argv) == 1:
        print("Received %d inputs instead of %d" % (len(sys.argv), 9))
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


# Returns the R2-score cross-validated N times
def cv_r2score(solver, N, X, y):
    r2_train = np.zeros(N)
    r2_test = np.zeros(N)
    i = 0
    kf = KFold(n_splits=N)
    for train, test in kf.split(X, y):
        x_train = X[train]
        x_test = X[test]
        y_train = y[train]
        y_test = y[test]
        fit_solver = solver.fit(x_train, y_train)
        pred_solver_train = fit_solver.predict(x_train)
        pred_solver_test = fit_solver.predict(x_test)
        r2_test[i] = r2_score(y_test, pred_solver_test)
        r2_train[i] = r2_score(y_train, pred_solver_train)
        i += 1
    return r2_train, r2_test


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
    imp = SimpleImputer(strategy='most_frequent')
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    # Data normalization
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    print('\nDATA NORMALIZATION REPORT')
    print('Mean of X_train:', x_train.mean())
    print('Std of X_train:', x_train.std())

    # PCA decomposition
    if options.use_pca:
        print("Using PCA.")
        pca = PCA(n_components=options.pca_belief_ratio, whiten=True).fit(x_train)
        x_train = pca.transform(x_train)

    print_time_since_checkpoint(checkpoint_time)
    print('\nAPPLY LEARNING METHOD')
    assert(np.isfinite(x_train).all())
    params = {'n_estimators': 200, 'max_depth': 3,
              'learning_rate': 0.1, 'loss': 'huber'}
    reg = GradientBoostingRegressor(**params).fit(x_train, y_train.ravel())
    print("Training Coefficient of Determination (R^2): %0.4f" %
          reg.score(x_train, y_train.ravel()))

    x_test = imp.transform(x_test)
    assert(np.isfinite(x_test).all())
    x_test = scaler.transform(x_test)
    if options.use_pca:
        x_test = pca.transform(x_test)
    y_pred = reg.predict(x_test)

    # Check performance
    r2score_train, r2score_test = \
        cv_r2score(reg, options.n_folds, x_train, y_train.ravel())
    print('Training error:', r2score_train, '\nTest error:', r2score_test)
    print_time_since_checkpoint(checkpoint_time)
    
    # Write prediction into solution.csv
    y_pred = [''.join(str(y) for y in x) for x in y_pred]
    id = [float(i) for i in range(0, len(y_pred))]
    df = pd.DataFrame({'id': id, 'y': y_pred})
    df.to_csv(options.predict, index=False)
    print('Prediction written to solution.csv')
    print_time_since_checkpoint(start_time)
    
    
if __name__ == '__main__':
    main()
