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
    
    if time.time()-checkpoint_time<60:
        print('>>> ', round(time.time()-checkpoint_time, 3), 's')
    else:
        print('>>> ', math.floor((time.time()-checkpoint_time)/60),
              'min', round(math.fmod(time.time()-checkpoint_time, 60)), 's')


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
    pca = PCA(n_components=0.999999, whiten=True).fit(x_train)
    x_train = pca.transform(x_train)

    print_time_since_checkpoint(checkpoint_time)
    print('\nAPPLY LEARNING METHOD')
    assert(np.isfinite(x_train).all())
    alphas = np.logspace(-4, 0, num=7)
    reg = RidgeCV(alphas, fit_intercept=True, cv=10).fit(x_train, y_train)
    print("Training Coefficient of Determination (R^2): %0.4f" %
          reg.score(x_train, y_train))

    x_test = scaler.transform(x_test)
    x_test[~np.isfinite(x_test)] = 0.
    x_test = pca.transform(x_test)
    y_pred = reg.predict(x_test)

    # Write prediction into solution.csv
    sol = pd.read_csv('sample.csv', header=None)
    z, s = np.shape(sol)
    for i in range(z - 1):
        sol.iat[i + 1, 1] = y_pred[i]
    sol.to_csv('solution.csv', index=False, header=None)
    print('Prediction written to solution.csv')
    print_time_since_checkpoint(start_time)
    
    
if __name__ == '__main__':
    main()
