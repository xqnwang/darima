#!/bin/bash -l
#SBATCH -J auto_arima
#SBATCH -n 32 # Number of cores
#SBATCH -t 10-00:00 # Runtime in D-HH:MM
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=xiaoqianwang@buaa.edu.cn


# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#echo Run the DARIMA model from $DIR

cd $DIR/..

Rscript  auto_arima.R  > auto_arima.out 2> auto_arima.log
