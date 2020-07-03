#! /usr/bin/env python3

# Python does not have a good function for automatic ARIMA modeling.
# We implement this via calling R code directly, provided that
# R package `forecast` and python package `rpy2` are both installed.
# FIXME: write a native `sarima2ar_model` R function.

import os, zipfile, pathlib
import numpy as np
import pandas as pd
import functools
import warnings

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import rpy2

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType


##--------------------------------------------------------------------------------------
# R version
##--------------------------------------------------------------------------------------
#robjects.r.source("~/xiaoqian-darima/darima//R/sarima2ar_model.R", verbose=False)
# robjects.r.source(exprs=zipfile.ZipFile(pathlib.Path(__file__).parents[1]).open("darima/R/test_fun.R").read().decode("utf-8"), verbose=True)
# ## robjects.r.source("darima/R/test_fun.R", verbose=False)
# test_fun=robjects.r['test.fun']

# def test_py(dat):
#     x = dat["v"].values
#     numpy2ri.activate()
#     out_r = test_fun(robjects.FloatVector(x))
#     numpy2ri.deactivate()
#     out_rb = robjects.FloatVector(out_r)
#     out_mean = pd.DataFrame(np.array(out_rb).reshape(1, 1), columns=['mvalue'])
#     par_id = pd.DataFrame(np.arange(1).reshape(1, 1), columns=['id'])
#     out = pd.concat([out_mean, par_id],1)
#     return(out)


sarima2ar_model_rcode = zipfile.ZipFile(pathlib.Path(__file__).parents[1]).open("darima/R/sarima2ar_model.R").read().decode("utf-8")
robjects.r.source(exprs=rpy2.rinterface.parse(sarima2ar_model_rcode), verbose=False)
sarima2ar_model=robjects.r['sarima2ar']

##--------------------------------------------------------------------------------------
# Python version - Simplified output version
##--------------------------------------------------------------------------------------
def darima_model(sample_df, Y_name, period = 1, tol = 500,
         order = [0,0,0], seasonal = [0,0,0],
         max_p = 5, max_q = 5, max_P = 2, max_Q = 2,
         max_order = 5, max_d = 2, max_D = 1,
         allowmean = True, allowdrift = True, method = "",
         approximation = False, stepwise = True,
         parallel = False, num_cores = 2):
    '''
    Fit ARIMA model for subseries and transform it into AR representation
    # sample_df: Pandas DataFrame
    '''

    # Get series data as numpy array (pdf -> numpy array)
    #--------------------------------------
    x_train = sample_df[Y_name].values
    n = len(sample_df) # length of subseries N_k

    # Fit model
    #--------------------------------------
    dfitted = sarima2ar_model(robjects.FloatVector(x_train), period = period, tol = tol,
                     order = robjects.IntVector(order), seasonal = robjects.IntVector(seasonal),
                     max_p = max_p, max_q = max_q, max_P = max_P, max_Q = max_Q,
                     max_order = max_order, max_d = max_d, max_D = max_D,
                     allowmean = allowmean, allowdrift = allowdrift, method = method,
                     approximation = approximation, stepwise = stepwise,
                     parallel = parallel, num_cores = num_cores)

    # Extract returns
    #--------------------------------------
    coef = robjects.FloatVector(dfitted.rx2("coef"))
    sigma2 = robjects.FloatVector(dfitted.rx2("sigma2"))

    # R object to python object
    #--------------------------------------
    p = tol + 2
    coef = np.array(coef).reshape(1, p) # 1-by-p
    sigma2 = np.array(sigma2).reshape(1, 1) # 1-by-1

    # Calculate Sig_inv and Sig_invMcoef
    #--------------------------------------
    Sig_inv_value = n/sigma2 # 1-by-1
    Sig_invMcoef = Sig_inv_value*coef # 1-by-p

    # Assign par_id
    #--------------------------------------
    par_id = sample_df["partition_id"].values[0]
    par_id = np.array(par_id).reshape(1, 1)

    # Out
    #--------------------------------------
    ar_coef_name = ['c0', 'c1'] + ['pi' + str(i+1) for i in np.arange(tol)]

    out_np = np.concatenate((par_id, Sig_inv_value, Sig_invMcoef),1) # 1-by-(2+p)
    out = pd.DataFrame(out_np,
                       columns=pd.Index(["par_id", "Sig_inv_value"] + ar_coef_name))

    if out.isna().values.any():
        warnings.warn("NAs appear in the final output")

    return out


##--------------------------------------------------------------------------------------
# Python version - Standard output version
##--------------------------------------------------------------------------------------
#def darima_model(sample_df, Y_name, period = 1, tol = 500,
#         order = [0,0,0], seasonal = [0,0,0],
#         max_p = 5, max_q = 5, max_P = 2, max_Q = 2,
#         max_order = 5, max_d = 2, max_D = 1,
#         allowmean = True, allowdrift = True, method = "",
#         approximation = False, stepwise = True,
#         parallel = False, num_cores = 2):
#    '''
#    Fit ARIMA model for subseries and transform it into AR representation
#    # sample_df: Pandas DataFrame
#    '''
#
#    # Get series data as numpy array (pdf -> numpy array)
#    #--------------------------------------
#    x_train = sample_df[Y_name].values
#    n = len(sample_df) # length of subseries N_k
#
#    # Creating rpy2 vectors
#    # robjects.FloatVector(x_train)
#    # robjects.IntVector(order)
#
#    # Fit model
#    #--------------------------------------
#    dfitted = sarima2ar_model(robjects.FloatVector(x_train), period = period, tol = tol,
#                     order = robjects.IntVector(order), seasonal = robjects.IntVector(seasonal),
#                     max_p = max_p, max_q = max_q, max_P = max_P, max_Q = max_Q,
#                     max_order = max_order, max_d = max_d, max_D = max_D,
#                     allowmean = allowmean, allowdrift = allowdrift, method = method,
#                     approximation = approximation, stepwise = stepwise,
#                     parallel = parallel, num_cores = num_cores)
#
#    # Extract returns
#    #--------------------------------------
#    coef = robjects.FloatVector(dfitted.rx2("coef"))
#    var_coef = robjects.FloatVector(dfitted.rx2("var.coef"))
#
#    # R object to python object
#    #--------------------------------------
#    p = tol + 2
#    coef = np.array(coef).reshape(p, 1) # p-by-1
#    var_coef = np.array(var_coef) # p-by-p
#
#    # Calculate Sig_inv and Sig_invMcoef
#    #--------------------------------------
#    Sig_inv = np.linalg.inv(var_coef)*n # p-by-p
#    Sig_invMcoef = Sig_inv.dot(coef) # p-by-1
#
#    # Assign par_id
#    #--------------------------------------
#    # According to subseries_id
#    # par_id_num = sample_df["partition_id"].values[0]
#    # par_id = pd.DataFrame(np.array([par_id_num]*p).reshape(p, 1), columns=['par_id']) # p-by-1
#
#    # According to coef_id
#    par_id = pd.DataFrame(np.arange(p).reshape(p, 1), columns=['par_id']) # p-by-1
#
#    # Out
#    #--------------------------------------
#    ar_coef_name = ['c0', 'c1'] + ['pi' + str(i+1) for i in np.arange(tol)]
#
#    out_np = np.concatenate((coef, Sig_invMcoef, Sig_inv),1) # p-by-(2+p)
#    out_pdf = pd.DataFrame(out_np,
#                           columns=pd.Index(["coef", "Sig_invMcoef"] + ar_coef_name))
#    out = pd.concat([par_id, out_pdf],1) # p-by-(3+p)
#
#    if out.isna().values.any():
#        warnings.warn("NAs appear in the final output")
#
#    return out