#! /usr/bin/env python3

import os, zipfile, pathlib
import numpy as np
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

#import rpy2.robjects as robjects
#from rpy2.robjects import numpy2ri
#from rpy2.robjects.packages import importr
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.conversion import localconverter
#import rpy2


##--------------------------------------------------------------------------------------
# R version
##--------------------------------------------------------------------------------------
#dlsa_rcode = zipfile.ZipFile(pathlib.Path(__file__).parents[1]).open("darima/R/dlsa.R").read().decode("utf-8")
#robjects.r.source(exprs=rpy2.rinterface.parse(dlsa_rcode), verbose=False)
#dlsa_comb=robjects.r['dlsa.comb']

##--------------------------------------------------------------------------------------
# Python version1 - by rpy2
##--------------------------------------------------------------------------------------
#def dlsa_mapreduce(model_mapped_sdf, sample_size):
#    '''
#    MapReduce for partitioned data with given model
#    Calculate global estimator
#    '''
#    model_mapped_pdf = model_mapped_sdf.toPandas()
#
#    # From pandas to R
#    with localconverter(robjects.default_converter + pandas2ri.converter):
#        model_mapped_r = robjects.conversion.py2rpy(model_mapped_pdf)
#
#    # DLSA
#    out_r = dlsa_comb(model_mapped_r, sample_size)
#
#    # From R to pandas
#    with localconverter(robjects.default_converter + pandas2ri.converter):
#        out = robjects.conversion.rpy2py(out_r)
#
#    return out


##--------------------------------------------------------------------------------------
# Python version - Simplified output version
##--------------------------------------------------------------------------------------
def dlsa_mapreduce(model_mapped_sdf, sample_size):
    '''
    MapReduce for partitioned data with given model
    Calculate global estimator
    '''
    # Spark data frame to Pandas data frame
    #--------------------------------------
    model_mapped_sdf = model_mapped_sdf.drop('par_id')
    model_mapped_pdf = model_mapped_sdf.toPandas()

    # Sum of each column
    #--------------------------------------
    p = model_mapped_pdf.shape[1] - 1
    model_mapped_sum = np.array(model_mapped_pdf.apply(lambda x: x.sum())).reshape(1, p+1)

    # Extract required results
    #--------------------------------------
    Sig_inv_sum_value = np.array(model_mapped_sum[:,0]).reshape(1, 1) # 1-by-1
    Sig_invMcoef_sum = np.array(model_mapped_sum[:,1:]).reshape(p, 1) # p-by-1

    # Generate diag according Sig_inv_sum_value
    #--------------------------------------
    Sig_inv_sum_inv = 1/Sig_inv_sum_value * np.identity(p) # p-by-p

    # Get Theta_tilde and Sig_tilde
    #--------------------------------------
    Theta_tilde = Sig_inv_sum_inv.dot(Sig_invMcoef_sum) # p-by-1
    Sig_tilde = Sig_inv_sum_inv*sample_size # p-by-p
    
    # Out
    #--------------------------------------
    out = pd.DataFrame(np.concatenate((Theta_tilde, Sig_tilde), 1),
                       columns= ["Theta_tilde"] + model_mapped_sdf.columns[1:])
    
    return out


##--------------------------------------------------------------------------------------
# Python version - Standard output version
##--------------------------------------------------------------------------------------
#def dlsa_mapreduce(model_mapped_sdf, sample_size):
#    '''
#    MapReduce for partitioned data with given model
#    Calculate global estimator
#    '''
#    ##----------------------------------------------------------------------------------------
#    ## MERGE
#    ##----------------------------------------------------------------------------------------
#    groupped_sdf = model_mapped_sdf.groupby("par_id")
#    groupped_sdf_sum = groupped_sdf.sum(*model_mapped_sdf.columns[1:]) #TODO: Error with Python < 3.7 for > 255 arguments. Location 0 is 'par_id'
#    groupped_pdf_sum = groupped_sdf_sum.toPandas().sort_values("par_id")
#    
#    p = groupped_pdf_sum.shape[0]
#    
#    if groupped_pdf_sum.shape[0] == 0: # bad chunked models
#        
#        raise Exception("Zero-length grouped pandas DataFrame obtained, check the input.")
#    
#    else:
#        
#        # Extract required results
#        #--------------------------------------
#        Sig_invMcoef_sum = groupped_pdf_sum.iloc[:,2] # p-by-1
#        Sig_inv_sum = groupped_pdf_sum.iloc[:,3:] # p-by-p
#        
#        Sig_inv_sum_inv = np.linalg.inv(Sig_inv_sum) # p-by-p
#        
#        # Get Theta_tilde and Sig_tilde
#        #--------------------------------------
#        Theta_tilde = Sig_inv_sum_inv.dot(Sig_invMcoef_sum) # p-by-1
#        Sig_tilde = Sig_inv_sum_inv*sample_size # p-by-p
#        
#        # Reshape
#        #--------------------------------------
#        Theta_tilde = np.array(Theta_tilde).reshape(p, 1)
#        Sig_tilde = np.array(Sig_tilde).reshape(p, p)
#        
#        out = pd.DataFrame(np.concatenate((Theta_tilde, Sig_tilde), 1),
#                           columns= ["Theta_tilde"] + model_mapped_sdf.columns[3:])
#    
#    return out
