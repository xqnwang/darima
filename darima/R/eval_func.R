#' Calculate the forecasting accuracy by considering MASE, sMAPE and MSIS as the measures 
#' 
#' Calculate the forecasting accuracy by considering MASE, sMAPE and MSIS as the measures 
#' 
#' @param x A vector or ts of the observed time-series values
#' @param xx A vector or ts of the future observations
#' @param period Length of the seasonal period
#' @param pred Point forecasts.
#' @param lower Lower limits for prediction intervals
#' @param upper Upper limits for prediction intervals
#' @param level Confidence level for prediction intervals
#' 
#' @return A list with the elements having the following structure
#' \item{mase}{A vector of the calculated MASE scores}
#' \item{smape}{A vector of the calculated sMAPE scores}
#' \item{msis}{A vector of the calculated MSIS scores}
#' 
#' @export
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