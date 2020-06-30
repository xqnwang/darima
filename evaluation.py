#! /usr/bin/env python3

# FIXME: write a native `eval_func` R function.

import os
import numpy as np
import pandas as pd
import functools
import warnings

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType


##--------------------------------------------------------------------------------------
# R version
##--------------------------------------------------------------------------------------
# robjects.r.source("~/xiaoqian-darima/darima//R/eval_func.R", verbose=False)
robjects.r.source(os.path.dirname(os.path.abspath(__file__)) + "/R/eval_func.R", verbose=False)
eval_func=robjects.r['eval_func']

##--------------------------------------------------------------------------------------
# Python version
##--------------------------------------------------------------------------------------
def model_eval(x, xx, period, 
               pred, lower, upper, level = 95):
    '''
    Evaluation
    '''
    
    # Get series data as numpy array (pdf -> numpy array)
    #--------------------------------------
    h = len(xx)
    x = x
    xx = xx
    pred = pred
    lower = lower
    upper = upper
    
    # Creating rpy2 vectors
    # robjects.FloatVector(x)
    
    # Forecasting
    #--------------------------------------   
    eval_result = eval_func(x = robjects.FloatVector(x), 
                      xx = robjects.FloatVector(xx), 
                      period = period,
                      pred = robjects.FloatVector(pred),
                      lower = robjects.FloatVector(lower),
                      upper = robjects.FloatVector(upper),
                      level = level)
    
    # Extract returns
    #--------------------------------------    
    mase = robjects.FloatVector(eval_result.rx2("mase"))
    smape = robjects.FloatVector(eval_result.rx2("smape"))
    msis = robjects.FloatVector(eval_result.rx2("msis"))
    
    # R object to python object
    #--------------------------------------
    mase = np.array(mase).reshape(h, 1) # h-by-1
    smape = np.array(smape).reshape(h, 1) # h-by-1
    msis = np.array(msis).reshape(h, 1) # h-by-1
    
    # Out
    #--------------------------------------        
    out_np = np.concatenate((mase, smape, msis),1) # h-by-3
    out_pdf = pd.DataFrame(out_np,
                           columns=pd.Index(["mase", "smape", "msis"]))
    out = out_pdf
    
    if out.isna().values.any():
        warnings.warn("NAs appear in the final output")
    
    return out