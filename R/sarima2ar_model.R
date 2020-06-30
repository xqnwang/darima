suppressPackageStartupMessages(require("forecast"))
suppressPackageStartupMessages(require("polynom"))

ar_coefficients <- function(ar = 0, d = 0L, ma = 0, 
                            sar = 0, D = 0L, sma = 0, 
                            mean = 0, drift = 0, 
                            m = 1L, tol = 500L) {
  mu <- mean
  dft <- drift
  
  # non-seasonal AR
  ar <- polynomial(c(1, -ar)) * polynomial(c(1, -1))^d
  
  # seasonal AR
  if (m > 1) {
    P <- length(sar)
    seasonal_poly <- numeric(m * P)
    seasonal_poly[m * seq(P)] <- sar
    sar <- polynomial(c(1, -seasonal_poly)) * polynomial(c(1, rep(0, m - 1), -1))^D
  }
  else {
    sar <- 1
  }
  
  # non-seasonal MA
  ma <- polynomial(c(1, ma))
  
  # seasonal MA
  if (m > 1) {
    Q <- length(sma)
    seasonal_poly <- numeric(m * Q)
    seasonal_poly[m * seq(Q)] <- sma
    sma <- polynomial(c(1, seasonal_poly))
  }
  else {
    sma <- 1
  }
  
  # pie
  n <- tol
  theta <- -c(coef(ma * sma))[-1]
  if (length(theta) == 0L) {
    theta <- 0
  }
  phi <- -c(coef(ar * sar)[-1], numeric(n))
  q <- length(theta)
  pie <- c(numeric(q), 1, numeric(n))
  for (j in seq(n))
    pie[j + q + 1L] <- -phi[j] + sum(theta * pie[(q:1L) + j])
  pie <- pie[(0L:n) + q + 1L]
  pie <- head(pie, (tol+1)) 
  pie <- -pie[-1]
  
  c0 <- mu * (1 - sum(pie)) + dft * (t(seq_len(tol)) %*% pie)
  c1 <- dft * (1 - sum(pie))
  
  # y_t = c0 + c1 * t + pie_1 * y_{t-1} + ... + pie_tol * y_{t-tol} + epsilon_t
  coef <- `names<-` (c(c0, c1, pie), 
                     c("beta0", "beta1", paste("ar", sep = "", seq_len(tol))))
  
  return(coef)
}



sarima2ar <- function(x, period = 1, tol = 500L, 
                      order = c(0L, 0L, 0L), seasonal = c(0L, 0L, 0L), 
                      max.p = 5, max.q = 5, max.P = 2, max.Q = 2, 
                      max.order = 5, max.d = 2, max.D = 1, 
                      allowmean = TRUE, allowdrift = TRUE, method = NULL, 
                      approximation = (length(x) > 150 | period > 12), 
                      stepwise = TRUE, parallel = FALSE, num.cores = 2){
  
  # create time-series objects
  x <- ts(x, frequency = period)
  
  # sarima model
  if (sum(order) == 0L & sum(seasonal) == 0L){
    # no pre-defined order and seasonal order
    fit <- forecast::auto.arima(x, allowdrift = allowdrift, allowmean = allowmean, 
                                max.p = max.p, max.q = max.q, max.P = max.P, max.Q = max.Q, 
                                max.order = max.order, max.d = max.d, max.D = max.D, 
                                method = method, stepwise = stepwise, 
                                approximation = approximation, 
                                parallel = parallel, num.cores = num.cores) 
  } else {
    fit <- forecast::Arima(x, order = order, seasonal = seasonal, method = method, 
                           include.mean = allowmean, include.drift = allowdrift)
  }
  sigma2 <- fit$sigma2
  
  # extract coefficents of sarima model
  isEmpty <- function(x) {
    return(length(x)==0)
  }
  d <- fit$arma[6]
  D <- fit$arma[7]
  m <- fit$arma[5]
  mu <- fit$coef[names(fit$coef) == "intercept"]
  if (isEmpty(mu))
    mu <- 0
  dft <- fit$coef[names(fit$coef) == "drift"]
  if (isEmpty(dft))
    dft <- 0
  phi <- fit$coef[substring(names(fit$coef), 1, 2) == "ar"]
  theta <- fit$coef[substring(names(fit$coef), 1, 2) == "ma"]
  Phi <- fit$coef[substring(names(fit$coef), 1, 3) == "sar"]
  Theta <- fit$coef[substring(names(fit$coef), 1, 3) == "sma"]
  
  # invert sarima to ar representation (ar coefficients)
  ar.coef <- ar_coefficients(ar = phi, d = d, ma = theta, 
                             sar = Phi, D = D, sma = Theta, 
                             mean = mu, drift = dft, 
                             m = m, tol = tol)
  
  # var.coef
  vc_i <- diag(length(ar.coef))
  colnames(vc_i) <- rownames(vc_i) <- names(ar.coef)
  vc <- sigma2 * vc_i
  
  # return
  return(list(coef = ar.coef, var.coef = vc))
  
}


