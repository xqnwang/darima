#! /usr/bin/env python3

import os
import numpy as np
import pandas as pd

# import rpy2.robjects as robjects
# from rpy2.robjects import numpy2ri

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

def dlsa_mapreduce(model_mapped_sdf, sample_size):
    '''
    MapReduce for partitioned data with given model
    Calculate global estimator
    '''
    # mapped_pdf = model_mapped_sdf.toPandas()
    ##----------------------------------------------------------------------------------------
    ## MERGE
    ##----------------------------------------------------------------------------------------
    groupped_sdf = model_mapped_sdf.groupby("par_id")
    groupped_sdf_sum = groupped_sdf.sum(*model_mapped_sdf.columns[1:]) #TODO: Error with Python < 3.7 for > 255 arguments. Location 0 is 'par_id'
    groupped_pdf_sum = groupped_sdf_sum.toPandas().sort_values("par_id")
    
    p = groupped_pdf_sum.shape[0]
    
    if groupped_pdf_sum.shape[0] == 0: # bad chunked models
        
        raise Exception("Zero-length grouped pandas DataFrame obtained, check the input.")
    
    else:
        
        Sig_invMcoef_sum = groupped_pdf_sum.iloc[:,2] # p-by-1
        Sig_inv_sum = groupped_pdf_sum.iloc[:,3:] # p-by-p
        
        Sig_inv_sum_inv = np.linalg.inv(Sig_inv_sum) # p-by-p
        
        Theta_tilde = Sig_inv_sum_inv.dot(Sig_invMcoef_sum) # p-by-1
        Sig_tilde = Sig_inv_sum_inv*sample_size # p-by-p
        
        # reshape
        Theta_tilde = np.array(Theta_tilde).reshape(p, 1)
        Sig_tilde = np.array(Sig_tilde).reshape(p, p)
        
        out = pd.DataFrame(np.concatenate((Theta_tilde, Sig_tilde), 1),
                           columns= ["Theta_tilde"] + model_mapped_sdf.columns[3:])
    
    return out
