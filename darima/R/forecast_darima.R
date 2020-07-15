# Forecasting according to the trained DARIMA model
suppressPackageStartupMessages(require("forecast"))
suppressPackageStartupMessages(require("quantmod"))
suppressPackageStartupMessages(require("magrittr"))

predict.ar <- function(Theta, sigma2, x, n.ahead = 1L, se.fit = TRUE){
  # x: ts$x
  # return fitted values and forecasts
  
  if (n.ahead < 1L) 
    stop("'n.ahead' must be at least 1")
  if (missing(x))
    stop("argument 'x' is missing")
  if (!is.ts(x))
    stop("'x' must be a time series")
  
  h <- n.ahead
  n <- length(x)
  tspx <- tsp(x)
  st <- tspx[2]
  dt <- deltat(x)
  coef <- Theta
  p <- length(coef) - 2 # ar order
  X <- cbind(1, seq_len(n), quantmod::Lag(as.vector(x), k = 1:p))
  
  # fitted value
  fits <- (X %*% coef) %>% as.vector() %>% 
    ts(frequency = tspx[3], start = tspx[1])
  res <- x - fits
  
  # forecasts
  newdata <- x
  tsp(newdata) <- NULL
  class(newdata) <- NULL
  y <- c(newdata, rep.int(0, h))
  for (i in seq_len(h)) y[n + i] <- sum(coef * 
                                          c(1, n + i, y[n + i - seq_len(p)])) 
  pred <- y[n + seq_len(h)]
  pred <- ts(pred, frequency = tspx[3], start = st + dt)
  
  # se.fit
  if (se.fit){
    psi <- .Call(stats:::C_ar2ma, tail(coef, p), h - 1L)
    vars <- cumsum(c(1, psi^2))
    se <- sqrt(sigma2 * vars)[seq_len(h)]
    se <- ts(se, frequency = tspx[3], start = st + dt)
    list(fitted = fits, residuals = res, pred = pred, se = se)
  } else {
    list(fitted = fits, residuals = res, pred = pred)
  }
  
}


forecast.darima <- function(Theta, sigma2, x, period, h = 1L, level = c(80, 95), fan = FALSE,
                            lambda = NULL, biasadj = FALSE){
  x <- ts(x, frequency = period)
  pred <- predict.ar(Theta, sigma2, x, n.ahead = h)
  if (fan) {
    level <- seq(51, 99, by = 3)
  }   else {
    if (min(level) > 0 & max(level) < 1) {
      level <- 100 * level
    }
    else if (min(level) < 0 | max(level) > 99.99) {
      stop("Confidence limit out of range")
    }
  }
  nint <- length(level)
  lower <- matrix(NA, ncol = nint, nrow = length(pred$pred))
  upper <- lower
  for (i in 1:nint) {
    qq <- qnorm(0.5 * (1 + level[i]/100))
    lower[, i] <- pred$pred - qq * pred$se
    upper[, i] <- pred$pred + qq * pred$se
  }
  colnames(lower) <- colnames(upper) <- paste(level, "%", sep = "")
  
  if (!is.null(lambda)) {
    pred$pred <- InvBoxCox(pred$pred, lambda, biasadj, list(level = level, 
                                                            upper = upper, lower = lower))
    lower <- InvBoxCox(lower, lambda)
    upper <- InvBoxCox(upper, lambda)
    fits <- InvBoxCox(pred$fitted, lambda)
    x <- InvBoxCox(x, lambda)
  }
  return(list(level = level, mean = pred$pred, se = pred$se, 
              lower = lower, upper = upper, fitted = pred$fitted, 
              residuals = pred$residuals))
  
}


