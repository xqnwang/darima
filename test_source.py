#! /usr/local/bin/python3.7

import findspark
findspark.init("/usr/lib/spark-current")

import pyspark

import os, sys, time
from datetime import timedelta

# from hurry.filesize import size
import pickle
import numpy as np
import pandas as pd
import string
from math import ceil

from pyspark.sql.types import *
from pyspark.sql import functions
from pyspark.sql.functions import udf, pandas_udf, PandasUDFType, monotonically_increasing_id

from model import test_fun, test_py

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# Set Executor Env
conf = pyspark.SparkConf().setAppName("Spark DARIMA App").setExecutorEnv('ARROW_PRE_0_15_IPC_FORMAT', '1')
spark = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()

# Enable Arrow-based columnar data transfers
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")
#spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "true")


#####
print("test begin:")
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))
schema_beta = StructType(
    [StructField('id', IntegerType(), True), 
    StructField('mvalue', DoubleType(), True)])
@pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
def mean_udf(v):
    return test_py(v)
df.groupby("id").apply(mean_udf).show()
print("test end.")