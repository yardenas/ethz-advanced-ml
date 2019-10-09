# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:01:27 2019

@author: oforster, slionar, yardas
@group : Mlcodebreakers
"""

import time
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


# Prints the time since the last checkpoint (stored in checkpoint_time)
def print_time_since_checkpoint(checkpoint_time):
    
    if time.time()-checkpoint_time < 60:
        print('>>> ', round(time.time()-checkpoint_time, 3), 's')
    else:
        print('>>> ', math.floor((time.time()-checkpoint_time) / 60),
              'min', round(math.fmod(time.time()-checkpoint_time, 60)), 's')


# returns the R2-score cross-validated N times
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
    
    # read data from csv
    x_train_pd = pd.read_csv('X_train.csv')
    x_train = x_train_pd.values[:, 1:]
    x_test_pd = pd.read_csv('X_test.csv')
    x_test = x_test_pd.values[:, 1:]
    y_train_pd = pd.read_csv('y_train.csv')
    y_train = y_train_pd.values[:, 1:]

    print('\nINSPECTION OF THE DATA')
    print('Size of X_train: ', np.shape(x_train))
    print('Size of X_test: ', np.shape(x_test))
    print('Size of y_train: ', np.shape(y_train))
    print_time_since_checkpoint(checkpoint_time)
    checkpoint_time = time.time()

    print('\nFEATURE ENGINEERING')
    print("Junk percentage: %0.3f%%" % (x_train[~np.isfinite(x_train)].shape[0]
          / x_train.size * 100))

    # Data normalization
    scaler = RobustScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    # After scaling, the mean is approximately 0. therefore, replace NaNs
    # with it.
    x_train[~np.isfinite(x_train)] = 0.
    print('\nDATA NORMALIZATION REPORT')
    print('Mean of X_train:', x_train.mean())
    print('Std of X_train:', x_train.std())

    # PCA decomposition
    pca = PCA(n_components=0.85, whiten=True).fit(x_train)
    x_train = pca.transform(x_train)
    print_time_since_checkpoint(checkpoint_time)
    print('\nAPPLY LEARNING METHOD')
    assert(np.isfinite(x_train).all())
    alphas = np.logspace(-5, 0, num=8)
    reg = RidgeCV(alphas, fit_intercept=True, cv=None).fit(x_train, y_train)
    print("Training Coefficient of Determination (R^2): %0.4f" %
          reg.score(x_train, y_train))

    x_test = scaler.transform(x_test)
    x_test[~np.isfinite(x_test)] = 0.
    x_test = pca.transform(x_test)
    y_pred = reg.predict(x_test)

    # Check performance
    r2score_train, r2score_test = \
        cv_r2score(reg, 7, x_train, y_train.ravel())
    print('Training error:', r2score_train, '\nTest error:', r2score_test)
    print_time_since_checkpoint(checkpoint_time)
    
    # write prediction into solution.csv
    sol = pd.read_csv('sample.csv', header=None)
    z, s = np.shape(sol)
    for i in range(z - 1):
        sol.iat[i + 1, 1] = y_pred[i]
    sol.to_csv('solution.csv', index=False, header=None)
    print('Prediction written to solution.csv')
    print_time_since_checkpoint(start_time)
    
    
if __name__ == '__main__':
    main()
