# `darima`

Distributed ARIMA Models implemented with Apache Spark

# Introduction

DARIMA is designed to facilitate forecasting ultra-long time series by utilizing the industry-standard MapReduce framework. The algorithm is developed on Spark platform and both Python as well as R interfaces.

See [`darima`](darima) for developed functions used for implementing DARIMA models.
- [`model.py`](darima/model.py) : Train ARIMA models for each subseries and convert the trained models into AR representations (Mapper).
- [`dlsa.py`](darima/dlsa.py) : Combine the local estimators obtained in Mapper by minimizing the global loss function (Reducer).
- [`forecast.py`](darima/forecast.py) : Forecast the next H observations by utilizing the combined estimators.
- [`evaluation.py`](darima/evaluation.py) : Calculate the forecasting accuracy in terms as MASE, sMAPE and MSIS.
- [`R`](darima/R) : R functions designed for modeling, combining and forecasting. [rpy2](https://pypi.org/project/rpy2/) is needed as an interface to use R from Python.

# System requirements

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
    - `pyspark >= 2.3.1`
    - `rpy2 >= 3.0.4`
    - `scikit-learn >= 0.21.2`
    - `numpy >= 1.16.3`
    - `pandas >= 0.23.4`
- `R >= 3.5.2`
    - `forecast >= 8.5`
    - `polynom = 1.3.9`
    - `dplyr >= 0.8.4`
    - `quantmod >= 0.4.13`
    - `magrittr >= 1.5`

# Usage

## DARIMA
Run the [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) code to forecast the time series of the GEFCom2017 by utilizing DARIMA.

```sh
  ./bash/run_darima.sh
```
or simply run 
```py
  PYSPARK_PYTHON=/usr/local/bin/python3.7 ARROW_PRE_0_15_IPC_FORMAT=1 spark-submit ./run_darima.py
```
**Note**: `ARROW_PRE_0_15_IPC_FORMAT=1` is added to instruct `PyArrow >= 0.15.0` to use the legacy IPC format with the older Arrow Java that is in Spark 2.3.x and 2.4.x.

## ARIMA
Run the R code to forecast the time series of the GEFCom2017 by utilizing the `auto.arima()` function (used for comparison).
```sh
  ./bash/auto_arima.sh
```
or simply run 
```r
  Rscript auto_arima.R
```

# References

- [Xiaoqian Wang](https://xqnwang.rbind.io), [Yanfei Kang](https://yanfei.site), [Rob J Hyndman](https://robjhyndman.com), & [Feng Li](http://feng.li/) (2022). Distributed ARIMA models for ultra-long time series (in press). International Journal of Forecasting. [*_Working Paper_*](https://arxiv.org/abs/2007.09577).

