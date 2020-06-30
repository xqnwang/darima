#! /bin/bash

# https://help.aliyun.com/document_detail/28124.html
# Fat EXECUTORS: one executor per node
# EC=16
# EM=30g

# Tiny EXECUTORS: one executor per core
EC=1
EM=2g
EXECUTORS=64

# MODEL_FILE
MODEL_DESCRIPTION=$1
MODEL_FILE=run_darima
OUTPATH=/home/student/.xiaoqian/darima/result/

# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#echo Run the DARIMA model from $DIR

cd $DIR
rm -rf darima.zip
zip -r darima.zip darima/ setup.py -x "**/__pycache__/*" ".git/*"

tic0=`date +"%Y-%m-%d-%T"`
tic=`date +%s`
PYSPARK_PYTHON=/usr/local/bin/python3.7 spark-submit  \
              --master yarn  \
              --driver-memory 10g  \
              --executor-memory ${EM}  \
              --executor-cores ${EC}  \
              --num-executors ${EXECUTORS} \
              --conf spark.rpc.message.maxSize=2000 \
              $DIR/${MODEL_FILE}.py \
	      > ${OUTPATH}${MODEL_DESCRIPTION}_${MODEL_FILE}.NE${EXECUTORS}.EC${EC}_${tic0}.out 2> ${OUTPATH}${MODEL_DESCRIPTION}_${MODEL_FILE}.NE${EXECUTORS}.EC${EC}_${tic0}.log

toc=`date +%s`
runtime=$((toc-tic))
echo ${MODEL_FILE}.NE${EXECUTORS}.EC${EC} finished, "Time used (s):" $runtime

exit 0;
