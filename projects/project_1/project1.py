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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def print_time_since_checkpoint(checkpoint_time): # prints the time since the last checkpoint (stored in checkpoint_time)
    
    if time.time()-checkpoint_time<60:
        print('>>> ', round(time.time()-checkpoint_time, 3), 's')
    else:
        print('>>> ', math.floor((time.time()-checkpoint_time)/60), 'min', round(math.fmod(time.time()-checkpoint_time,60)), 's')
        
        
def cv_r2score(solver, N, X, y, print_params=False): # returns the R2-score crossvalidated N times
    
    r2 = np.zeros(N)
    i = 0
    
    kf = KFold(n_splits = N) 
    for train, test in kf.split(X, y):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        fit_solver = solver.fit(X_train,y_train)
        pred_solver = fit_solver.predict(X_test)
        r2[i] = r2_score(y_test, pred_solver)
        i += 1
        if print_params:
            print(solver.best_params_ )
            
    return r2
    

def main():
    start_time = time.time()
    checkpoint_time = start_time
    
    # read data from csv
    X_train_pd = pd.read_csv('X_train.csv')
    X_train = X_train_pd.values[:,1:]
    X_test_pd = pd.read_csv('X_test.csv')
    X_test = X_test_pd.values[:,1:]
    y_train_pd = pd.read_csv('y_train.csv')
    y_train = y_train_pd.values[:,1:]
    
    
    print('\nINSPECTION OF THE DATA')
    print('Size of X_train: ', np.shape(X_train))
    print('Size of X_test: ', np.shape(X_test))
#    print('Size of y_train: ', np.shape(y_train))
    print_time_since_checkpoint(checkpoint_time)
    checkpoint_time = time.time()
    
    
    print('\nFEATURE ENGINEERING')
    # substitute nan by 0
    for i in range(np.shape(X_train)[0]):
        for k in range(np.shape(X_train)[1]):
            if(np.isnan(X_train[i,k])):
                X_train[i,k] = 0
                
    for i in range(np.shape(X_test)[0]):
        for k in range(np.shape(X_test)[1]):
            if(np.isnan(X_test[i,k])):
                X_test[i,k] = 0
    
    print_time_since_checkpoint(checkpoint_time)
    checkpoint_time = time.time()
    
    
    print('\nAPPLY LEARNING METHOD')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Ridge regression with best alpha
#    alpha = np.logspace(0,8,num=9)
#    print(alpha)
#    parameters = {'alpha':(alpha)}
#    reg = Ridge(fit_intercept=False)
#    reg_best = GridSearchCV(reg, parameters, cv=3)
#    fit_reg_best = reg_best.fit(X_train,y_train)
#    print(fit_reg_best.best_params_)
#    y_pred = fit_reg_best.predict(X_test)
    
    print('MLP with parameters: ')
    alpha = [0.00001,0.001]
    MLPlayers = [(200,100)]
    parameters = {'alpha':(alpha), 'hidden_layer_sizes':(MLPlayers)}
    mlp = MLPRegressor(solver='lbfgs')
    mlp_best = GridSearchCV(mlp, parameters, cv=3)
    fit_mlp_best = mlp_best.fit(X_train,y_train.ravel())
    print(fit_mlp_best.best_params_)
    y_pred = fit_mlp_best.predict(X_test)
    
    # check performance
#    r2score = cv_r2score(mlp_best, 5, X_train, y_train.ravel(), print_params=True)
#    print(r2score)
    
    print_time_since_checkpoint(checkpoint_time)
    
    # write prediction into solution.csv
    sol = pd.read_csv('sample.csv', header=None)
    z,s = np.shape(sol)
    for i in range(z-1):
        sol.iat[i+1,1] = y_pred[i]
    sol.to_csv('solution.csv', index = False, header=None)
    print('Prediction written to solution.csv')
    print_time_since_checkpoint(start_time)
    
    
if __name__ == '__main__':
    main()