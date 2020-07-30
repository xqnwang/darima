suppressPackageStartupMessages(require("forecast"))
suppressPackageStartupMessages(require("polynom"))

#' Compute the Autoregressive coefficients of a seasonal ARIMA model
#' 
#' Convert a SARIMA(p,d,q,P,D,Q) model to its AR representation and 
#' return the AR coefficients.
#' 
#' @param ar p vector of autoregressive coefficients in the SARIMA model.
#' @param d Number of differences in the SARIMA model.
#' @param ma q vector of moving average coefficients in the SARIMA model.
#' @param sar P vector of seasonal autoregressive coefficients in the SARIMA model.
#' @param D Number of seasonal differences in the SARIMA model.
#' @param sma Q vector of moving average coefficients in the SARIMA model.
#' @param mean The coefficient of the mean term in the SARIMA model.
#' @param drift The coefficient of the linear drift term in the SARIMA model.
#' @param m The length of the seasonal period.
#' @param tol A large AR order used to make the approximated AR model 
#' infinitely close to the true AR process.
#' 
#' @return A vector of the converted AR coefficients.
#' 
#' @examples 
#' ar <- c(0.2, 0.3, 0.5)
#' ma <- c(0.1, 0.4)
#' sar <- c(0.3, 0.2)
#' sma <- c(0.3)
#' d <- 2
#' D <- 1
#' mean <- 0.3
#' ar_coefficients(ar = ar, d = d, ma = ma, 
#'                 sar = sar, D = D, sma = sma, 
#'                 mean = mean, drift = drift, 
#'                 m = 12, tol = 100)
#' 
#' @export
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


#' Fit ARIMA model to univariate time series and return its AR representation
#' 
#' Automatic ARIMA modeling to a univariate time series and return its 
#' approximated AR representation.
#' 
#' See the \code{\link[forecast]{auto.arima}} function in the forecast package.
#' 
#' @param x A vector or ts of the observed time-series values.
#' @param period The length of the seasonal period.
#' @param tol A large AR order used to make the approximated AR model 
#' infinitely close to the true AR process.
#' @param order A specification of the non-seasonal part of the ARIMA 
#' model: the three components (p, d, q) are the AR order, the degree of 
#' differencing, and the MA order.
#' @param seasonal A specification of the seasonal part of the ARIMA 
#' model: the three components (P, D, Q) are the AR order, the degree of 
#' differencing required for a seasonally stationary series, and the MA order.
#' @param max.p Maximum value of p in automatic ARIMA modeling.
#' @param max.q Maximum value of q in automatic ARIMA modeling.
#' @param max.P Maximum value of P in automatic ARIMA modeling.
#' @param max.Q Maximum value of Q in automatic ARIMA modeling.
#' @param max.order Maximum value of p+q+P+Q if model selection is not stepwise 
#' in automatic ARIMA modeling.
#' @param max.d	Maximum number of non-seasonal differences in automatic 
#' ARIMA modeling.
#' @param max.D	Maximum number of seasonal differences in automatic 
#' ARIMA modeling.
#' @param allowmean	If TRUE, models with a non-zero mean are considered in 
#' automatic ARIMA modeling.
#' @param allowdrift If TRUE, models with drift terms are considered in 
#' automatic ARIMA modeling.
#' @param method Fitting method in automatic ARIMA modeling: 
#' maximum likelihood or minimize conditional sum-of-squares. 
#' The default (unless there are missing values) is to use 
#' conditional-sum-of-squares to find starting values, then maximum likelihood. 
#' Can be abbreviated.
#' @param approximation If TRUE, estimation is via conditional sums 
#' of squares and the information criteria used for model selection are 
#' approximated in automatic ARIMA modeling. The final model is still 
#' computed using maximum likelihood estimation. Approximation should be 
#' used for long time series or a high seasonal period to avoid excessive 
#' computation times.
#' @param stepwise If TRUE, will do stepwise selection (faster) in 
#' automatic ARIMA modeling. Otherwise, it searches over all models. 
#' Non-stepwise selection can be very slow, especially for seasonal models.
#' @param parallel If TRUE and stepwise = FALSE, then the specification search 
#' is done in parallel. This can give a significant speedup on mutlicore machines.
#' @param num.cores Allows the user to specify the amount of parallel processes 
#' to be used if parallel = TRUE and stepwise = FALSE. If NULL, then the number 
#' of logical cores is automatically detected and all available cores are used.
#' 
#' @return A list with the elements having the following structure
#' \item{coef}{A vector of the approximated AR coefficients.}
#' \item{sigma2}{The bias adjusted MLE of the innovations variance of the fitted 
#' ARIMA model.}
#' 
#' @examples 
#' sarima2ar(USAccDeaths, period = 12, tol = 100, method = "CSS")
#' 
#' @export
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
  sigma2 <- c(fit$sigma2)
  
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
  #vc_i <- diag(length(ar.coef))
  #colnames(vc_i) <- rownames(vc_i) <- names(ar.coef)
  #vc <- sigma2 * vc_i
  
  # return
  #return(list(coef = ar.coef, var.coef = vc))
  return(list(coef = ar.coef, sigma2 = sigma2))
  
}


