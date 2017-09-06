#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:06:15 2017

@author: Hideto Koizumi,
PhD Student at Wharton School

Purpose: conduct intervally-censored regression, as 'intreg' in Stata.
Note: 'survreg' in R is not general enough to take care of right and left-
    censored observations
    
Acknowlegements: Stata ado-file 'intreg', Python 'tobit' module
    "https://github.com/jamesdj/tobit", by jamesdj.
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats
from scipy.special import log_ndtr
from statsmodels.regression.linear_model import WLS
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sklearn.preprocessing as skl

def split_left_right_censored(x, y1, y2, cens, w): ## 'cens' has to be created
    counts = cens.value_counts()
    if -1 not in counts and 1 not in counts:
        warnings.warn("No censored observations; use regression methods for uncensored data")
    xs = []
    ys = []
    ws = []
#    w = [1]*len(y1) ## analytical weights
    ## check if w is inputted. If not, assign a vector of 1s
    try:
        w
    except NameError:
        var_exists = False
    else:
        var_exists = True
    if var_exists == False:
        w = [1]*len(y1)
        
    for value in [-1, 0, 1]:
        if value in counts:
            if value == -1:
                split = cens == value
                y_split = np.squeeze(y2[split].values)
                x_split = x[split]
                w_split = w[split]
            elif value == 1:
                split = cens == value
                y_split = np.squeeze(y1[split].values)
                x_split = x[split]
                w_split = w[split]
            elif value == 0:
                split = cens == value
                ys1 = np.squeeze(y1[split].values)
                ys2 = np.squeeze(y2[split].values)
                x_split = x[split] 
                w_split = w[split]
        else:
            y_split, x_split, w_split = None, None, None
        xs.append(x_split)
        ws.append(w_split)
        if value == -1 or value == 1:
            ys.append(y_split)        
    
    return xs, ys, ys1, ys2, ws


def tobit_neg_log_likelihood(xs, ys, ys1, ys2, ws, params):
    x_left, x_mid, x_right = xs
    w_left, w_mid, w_right = ws    
    y_left, y_right = ys

    b = params[:-1]
    # s = math.exp(params[-1])
    s = params[-1]

    to_cat = []

    cens2 = False
    if y_left is not None:
        cens2 = True
        left = w_left*scipy.stats.norm.logcdf((y_left - np.dot(x_left, b)) / s)
        to_cat.append(left)
    if y_right is not None:
        cens2 = True
#        right = scipy.stats.norm.logsf(y_right - np.dot(x_right, b) / s)
        right = w_right*np.log(1 - scipy.stats.norm.cdf((y_right - np.dot(x_right, b))/s))
        to_cat.append(right)
    if ys1 is not None and ys2 is not None:
        inter = w_mid*np.log(scipy.stats.norm.cdf((ys2 - np.dot(x_mid, b)) / s) - 
                       scipy.stats.norm.cdf((ys1 - np.dot(x_mid, b)) / s))
        to_cat.append(inter)    
    
#    if y_left is not None:
#        cens2 = True
#        left = w_left*scipy.stats.norm.logcdf((y_left - np.dot(x_left, b)) / (s / np.sqrt(w_left)))
#        to_cat.append(left)
#    if y_right is not None:
#        cens2 = True
##        right = scipy.stats.norm.logsf(y_right - np.dot(x_right, b) / s)
#        right = w_right*np.log(1 - scipy.stats.norm.cdf((y_right - np.dot(x_right, b))/(s/np.sqrt(w_right))))
#        to_cat.append(right)
#    if ys1 is not None and ys2 is not None:
#        inter = w_mid*np.log(scipy.stats.norm.cdf((ys2 - np.dot(x_mid, b)) / (s/np.sqrt(w_mid))) - 
#                       scipy.stats.norm.cdf((ys1 - np.dot(x_mid, b)) / (s/np.sqrt(w_mid))))
#        to_cat.append(inter)
        # mid_stats = (y_mid - np.dot(x_mid, b)) / s
        # mid = math.log(scipy.stats.norm.cdf(mid_stats) - scipy.stats.norm.cdf(
               # max(np.finfo('float').resolution, s))    
    else:
        mid_sum = 0    
    if cens2:
        # concat_stats = np.concatenate(to_cat, axis=0) / s
        log_cum_norm = np.concatenate(to_cat, axis=0)
        # log_cum_norm = left + right + inter # log_ndtr(concat_stats)
        cens_sum = log_cum_norm.sum()
    else:
        cens_sum = 0


        mid_sum = inter.sum()


    try:
        mid_sum
    except NameError:
        var_exists = False
    else:
        var_exists = True
    if var_exists == False:
        mid_sum = 0
    loglik = cens_sum + mid_sum

    return - loglik


class TobitModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.ols_coef_ = None
        self.ols_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None

    def fit(self, x, y1, y2, cens, w, verbose=False):
        """
        Fit a maximum-likelihood Tobit regression
        :param x: Pandas DataFrame (n_samples, n_features): Data
        :param y: Pandas Series (n_samples,): Target
        :param cens: Pandas Series (n_samples,): -1 indicates left-censored samples, 0 for uncensored, 1 for right-censored
        :param verbose: boolean, show info from minimization
        :return:
        """        
        x_copy = x.copy()
        if self.fit_intercept:
            x_copy = np.insert(x_copy, 0, 1, axis=1)
        else:
            x_copy = skl.scale(x_copy, with_mean=True, with_std=False, copy=False)

##		qui gen double `z' = cond(`y1'<.&`y2'<.,(`y1'+`y2')/2, /*
##		*/		 cond(`y1'<.,`y1',`y2')) `moff' if `doit'
        y = []
        counts = cens.value_counts() 
        for value in [-1, 0, 1]:
            if value in counts:
                if value == -1:
                    split = cens == value
                    y_l = np.squeeze(y2[split].values)
                    y.append(y_l)
                elif value == 1:
                    split = cens == value
                    y_r = np.squeeze(y1[split].values)
                    y.append(y_r)
                elif value == 0:
                    split = cens == value
                    y_int = np.squeeze((y1[split].values + y2[split].values)/2)
                    y.append(y_int)

        y = np.concatenate(y, axis=0)
        init_reg = WLS(y, x_copy, weights=w).fit()
        b0 = init_reg.params
        print(b0)
        y_pred = init_reg.predict(x_copy)
        resid = y - y_pred
        resid_var = np.var(resid)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, s0)
        xs, ys, ys1, ys2, ws = split_left_right_censored(x_copy, y1, y2, cens, w)
        result = minimize(lambda params: tobit_neg_log_likelihood(xs, ys, ys1, ys2, ws, params), params0,
                          jac=None, method='Powell', tol=0.000001,
                          options={'disp': verbose, 'maxiter':10000000, 'fatol':0.00000001})

        if verbose:
            print(result)
#        self.ols_coef_ = b0[1:]
#        self.ols_intercept = b0[0]
        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_ = result.x[1:-1]
        else:
            self.coef_ = result.x[:-1]
            self.intercept_ = 0
        self.sigma_ = result.x[-1]
        return self

    def predict(self, x):
        return self.intercept_ + np.dot(x, self.coef_)

    def score(self, x, y, scoring_function=mean_absolute_error):
        y_pred = np.dot(x, self.coef_)
        return scoring_function(y, y_pred)
