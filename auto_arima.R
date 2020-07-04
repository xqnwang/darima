#! /usr/bin/Rscript

################
### set path ###
################
path = "/home/student/.xiaoqian/darima"
setwd(path)

##################
### gefcom2017 ###
##################
load("data/gefcom2017.RData")

################
### packages ###
################
library(forecast)
library(magrittr)
library(polynom)
library(quantmod)
# parallel computation
library(parallel)
library(doParallel)
library(foreach)

###################################
### auto.arima (method = "CSS") ###
###################################
for (ncores in c(1,2,4,8,16,32)){
  print(paste0("Begin: ncores = ", ncores))
  t0_arima <- Sys.time()
  f_arima <- lapply(gefcom2017, function(lentry){
    t0 <- Sys.time()
    fit <- forecast::auto.arima(lentry$x, method = "CSS", 
                                stepwise = FALSE, parallel = TRUE, 
                                approximation = FALSE, 
                                num.cores = ncores)
    forec <- forecast(fit, h = lentry$h)
    t1 <- Sys.time()
    tt <- t1 - t0
    return(append(lentry, list(fit = fit, forec = forec, time = tt)))
  })
  assign(paste0("f_arima_NC", ncores), f_arima)
  t1_arima <- Sys.time()
  time_arima <- t1_arima - t0_arima
  print(paste0("End: ncores = ", ncores))
  print(time_arima)
  rm(t0_arima, t1_arima, time_arima, f_arima)
}

save.image("data/auto_arima_result.RData")
