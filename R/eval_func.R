eval_func <- function(x, xx, period, pred, lower, upper, level){
  freq <- period
  scaling <- mean(abs(diff(as.vector(x), freq)))
  mase <- abs(as.vector(xx) - as.vector(pred)) / scaling
  
  outsample <- as.numeric(xx)
  forecasts <- as.numeric(pred)
  smape <- (abs(outsample-forecasts)*200)/(abs(outsample)+abs(forecasts))
  
  # eg: level = 95
  alpha <- (100 - level)/100
  msis <- (upper - lower + 
             (2 / alpha) * (lower - xx) * (lower > xx) + 
             (2 / alpha) * (xx - upper) * (upper < xx)) / scaling
  
  return(list(mase = mase, smape = smape, msis = msis))
}