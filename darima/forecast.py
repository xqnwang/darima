#! /usr/local/bin/python3.7

# FIXME: write a native `forecast.ar` R function.

import os, zipfile, pathlib
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
# robjects.r.source("~/xiaoqian-darima/darima//R/forecast_darima.R", verbose=False)
## robjects.r.source(os.path.dirname(os.path.abspath(__file__)) + "/R/forecast_darima.R", verbose=False)

##robjects.r.source("darima/R/forecast_darima.R", verbose=False)
robjects.r.source(exprs=zipfile.ZipFile(pathlib.Path(__file__).parents[1]).open("darima/R/forecast_darima.R").read().decode("utf-8"), verbose=False)

forecast_darima=robjects.r['forecast.darima']

##--------------------------------------------------------------------------------------
# Python version
##--------------------------------------------------------------------------------------
def darima_forec(Theta, Sigma, x, period, h = 1, level = 95):
    '''
    Forecasting
    '''
    # Calculate sigma2 hat
    #--------------------------------------
    sigma2 = float(sum(Sigma.values.diagonal())/Sigma.shape[0])

    # Get series data as numpy array (pdf -> numpy array)
    #--------------------------------------
    Theta = Theta.values
    x = x.values

    # Creating rpy2 vectors
    # robjects.FloatVector(x)

    # Forecasting
    #--------------------------------------
    forec = forecast_darima(Theta = robjects.FloatVector(Theta), sigma2 = sigma2,
                     x = robjects.FloatVector(x), period = period,
                     h = h, level = level)

    # Extract returns
    #--------------------------------------
    pred = robjects.FloatVector(forec.rx2("mean"))
    lower = robjects.FloatVector(forec.rx2("lower"))
    upper = robjects.FloatVector(forec.rx2("upper"))

    # R object to python object
    #--------------------------------------
    pred = np.array(pred).reshape(h, 1) # h-by-1
    lower = np.array(lower).reshape(h, 1) # h-by-1
    upper = np.array(upper).reshape(h, 1) # h-by-1

    # Out
    #--------------------------------------
    out_np = np.concatenate((pred, lower, upper),1) # h-by-3
    out_pdf = pd.DataFrame(out_np,
                           columns=pd.Index(["pred", "lower", "upper"]))
    out = out_pdf

    if out.isna().values.any():
        warnings.warn("NAs appear in the final output")

    return out
