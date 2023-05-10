import pandas as pd
import numpy as np
import env
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from scipy import stats
import wrangle
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


def plot_residuals(y, yhat):
    residual = yhat - y
    plt.scatter(y , residual)
    plt.xlabel('x = calculatedfinishedsquarefeet')
    plt.ylabel('y = residual_baseline')
    plt.title('OLS linear model')
    plt.show()


def regression_errors(y, yhat):
    residual = yhat - y
    residual_squared = residual ** 2
    SSE = sum(residual_squared)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    MSE = SSE/len(y)
    RMSE2 = sqrt(mean_squared_error(yhat, y))
    
    return SSE, ESS, TSS, MSE, RMSE2


def baseline_mean_errors(y):
    baseline = y.mean()
   
    residual = baseline - y
    residual_squared = residual ** 2
    SSE = sum(residual_squared)
    ESS = sum((baseline - y)**2)   
    TSS = ESS + SSE   
    MSE = SSE/len(y) 
    RMSE2 = sqrt(mean_squared_error(baseline, y ))

    return SSE, ESS, TSS, MSE, RMSE2


def better_than_baseline(y, yhat):
    SSE, ESS, TSS, MSE, RMSE2 = regression_errors(y, yhat)
    SSE_baseline, ESS, TSS, MSE, RMSE2 = baseline_mean_errors(y)
    print(SSE) 
    print(SSE_baseline)
    if SSE_baseline > SSE:
        print('Model SSE is better than baseline SSE')
    else:
        print('baseline SSE is better than Model SSE')




