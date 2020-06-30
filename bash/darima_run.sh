#! /bin/bash

# https://help.aliyun.com/document_detail/28124.html

# Fat executors: one executor per node
# EC=16
# EM=30g

#zip -r dep.zip model.py dlsa.py forecast.py evaluation.py R/sarima2ar_model.R R/forecast_darima.R R/eval_func.R

MODEL_DESCRIPTION=$1

# Tiny executors: one executor per core
EC=1
EM=3g

# MODEL_FILE
MODEL_FILE=darima
OUTPATH=/home/student/xiaoqian-darima/darima/result/

# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# for i in 1 {4..100..4} # 1, 5, 10, 15, ... , 100

# for i in {256..4..-4}
tic0=`date +"%Y-%m-%d-%T"`
for executors in 30
do
    tic=`date +%s`
    PYSPARK_PYTHON=/usr/local/bin/python3.7 spark-submit  \
                  --master yarn  \
                  --driver-memory 50g  \
                  --executor-memory ${EM}  \
                  --num-executors ${executors}  \
                  --executor-cores ${EC}  \
                  --conf spark.rpc.message.maxSize=2000  
                  #--py-files $DIR/dep.zip  \
                  $DIR/../${MODEL_FILE}.py  \
                  > ${OUTPATH}${MODEL_DESCRIPTION}_${MODEL_FILE}.NE${executors}.EC${EC}_${tic0}.out 2> ${OUTPATH}${MODEL_DESCRIPTION}_${MODEL_FILE}.NE${executors}.EC${EC}_${tic0}.log
    toc=`date +%s`
    runtime=$((toc-tic))
    echo ${MODEL_FILE}.NE${executors}.EC${EC} finished, "Time used (s):" $runtime

done

exit 0;