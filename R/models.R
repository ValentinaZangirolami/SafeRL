# statistical models

library(R6)
library(data.table)
library(mgcv)
library(qgam)


# --- GAM regressor ---
GAMRegressor <- R6Class(
  "GAMRegressor",
  public = list(
    fit_obj = NULL,
    formula = NULL,
    
    fit = function(dt, y_col, x_terms, w = NULL) {
      stopifnot(is.data.table(dt))
      self$formula <- as.formula(paste(y_col, "~", paste(x_terms, collapse = " + ")))
      if (is.null(w)) w <- rep(1, nrow(dt))
      self$fit_obj <- mgcv::gam(self$formula, data = as.data.frame(dt), weights = w, method = "REML")
      invisible(self)
    },
    
    predict = function(dt) {
      stopifnot(!is.null(self$fit_obj))
      as.numeric(stats::predict(self$fit_obj, newdata = as.data.frame(dt), type = "response"))
    }
  )
)

# --- expectile GAM (V) ---
ExpectileGAM <- R6Class(
  "ExpectileGAM",
  public = list(
    tau = NULL,
    n_irls = NULL,
    base = NULL,
    x_terms = NULL,
    y_tmp = ".__y_tmp__",
    
    initialize = function(tau = 0.8, n_irls = 8) {
      stopifnot(tau > 0 && tau < 1)
      self$tau <- tau
      self$n_irls <- n_irls
      self$base <- GAMRegressor$new()
    },
    
    fit = function(dt, y_col, x_terms, init_w = NULL) {
      stopifnot(is.data.table(dt))
      self$x_terms <- x_terms
      d <- copy(dt)
      d[, (self$y_tmp) := get(y_col)]
      if (is.null(init_w)) init_w <- rep(1, nrow(d))
      w <- init_w
      
      for (k in seq_len(self$n_irls)) {
        self$base$fit(d, y_col = self$y_tmp, x_terms = x_terms, w = w)
        mu <- self$base$predict(d)
        diff <- d[[self$y_tmp]] - mu
        w <- ifelse(diff > 0, self$tau, 1 - self$tau) #expectile
        w <- pmax(w, 1e-6)
      }
      invisible(self)
    },
    
    predict = function(dt) {
      self$base$predict(dt)
    }
  )
)

# --- belief model: mean GAM + upper-tail quantile GAM ---
BeliefModel <- R6Class(
  "BeliefModel",
  public = list(
    mean_fit = NULL,
    upper_fit = NULL,
    tau = NULL,
    
    initialize = function(tau = 0.95) {
      stopifnot(tau > 0 && tau < 1)
      self$tau <- tau
    },
    
    fit = function(demand_dt, k_t = 8, k_past = 10, qgam_maxit = 400) {
      dt <- as.data.table(demand_dt)
      dt <- dt[!is.na(current_demand) & !is.na(past_demand)]
      
      
      f <- current_demand ~ s(hour, bs = "cc", k = k_t) + s(past_demand, k = k_past)
      
      # mgcv mean GAM
      self$mean_fit <- mgcv::gam(f, data = as.data.frame(dt), method = "REML")
      
      # qgam upper quantile
      self$upper_fit <- qgam::qgam(f, data = as.data.frame(dt), qu = self$tau)
      
      invisible(self)
    },
    
    predict = function(dt_like) {
      dt <- as.data.table(dt_like)
      est <- as.numeric(predict(self$mean_fit, newdata = as.data.frame(dt)))
      est_max <- as.numeric(predict(self$upper_fit, newdata = as.data.frame(dt)))
      list(est = est, est_max = est_max)
    }
  )
)

# --- policy: weighted MLE Gaussian---
WeightedGaussianPolicy <- R6Class(
  "WeightedGaussianPolicy",
  public = list(
    mu = NULL,
    mu_terms = NULL,
    sigma = NULL,
    deterministic = NULL,
    a_min = NULL,
    a_max = NULL,
    eps = 1e-6,
    
    initialize = function(mu_terms,
                          a_min = 0,
                          a_max = 10,
                          deterministic = TRUE) {
      self$mu <- GAMRegressor$new()
      self$mu_terms <- mu_terms
      self$deterministic <- deterministic
      self$a_min <- a_min
      self$a_max <- a_max
      self$sigma <- 1.0
    },
    
    fit = function(dt, w_col = "w", a_col = "a") {
      stopifnot(is.data.table(dt))
      
      a <- dt[[a_col]]
      w <- dt[[w_col]]
      
      w[!is.finite(w) | w < 0] <- 0
      if (sum(w) <= 0) w <- rep(1, nrow(dt))
      
      z <- (a - self$a_min) / (self$a_max - self$a_min)
      z <- pmin(pmax(z, self$eps), 1 - self$eps)
      
      eta <- qlogis(z)
      
      d <- copy(dt)
      d[, eta := eta]
      
      self$mu$fit(d, y_col = "eta", x_terms = self$mu_terms, w = w)
      
      eta_hat <- self$mu$predict(d)
      resid <- eta - eta_hat
      self$sigma <- sqrt(sum(w * resid^2) / sum(w))
      
      if (!is.finite(self$sigma) || self$sigma < 1e-6)
        self$sigma <- 1e-3
      
      invisible(self)
    },
    
    # actions
    act = function(dt_state) {
      eta_hat <- as.numeric(self$mu$predict(dt_state))
      
      if (!self$deterministic) {
        eta_hat <- rnorm(1, mean = eta_hat, sd = self$sigma)
      }
      
      # squash to bounds
      a <- self$a_min +
        (self$a_max - self$a_min) * plogis(eta_hat)
      
      as.numeric(a)
    }
  )
)



RewardLearner <- R6Class(
  "RewardLearner",
  public = list(
    gamma = NULL,
    Q = NULL,
    V = NULL,
    q_terms = NULL,
    v_terms = NULL,
    reward_col = NULL,
    mask_col = NULL,
    
    initialize = function(gamma, Q_model, V_model,
                          q_terms, v_terms,
                          reward_col = "r",
                          mask_col = "mask") {
      self$gamma <- gamma
      self$Q <- Q_model
      self$V <- V_model
      self$q_terms <- q_terms
      self$v_terms <- v_terms
      self$reward_col <- reward_col
      self$mask_col <- mask_col
    },
    
    fit = function(dt, n_iter = 30) {
      d <- as.data.table(copy(dt))
      if (!(self$mask_col %in% names(d))) d[, (self$mask_col) := 1.0]
      
      # ---- BOOTSTRAP Q ----
      d[, target0 := get(self$reward_col)]
      self$Q$fit(d, y_col = "target0", x_terms = self$q_terms)
      
      for (k in seq_len(n_iter)) {
        
        # ---- V update ----
        d[, q_val := self$Q$predict(.SD), .SDcols = c("x", "a", "hour")]
        self$V$fit(d, y_col = "q_val", x_terms = self$v_terms)
        
        # ---- Q update ----
        dt_next <- data.table(x = d[["x_next"]], hour = d[["hour_next"]])
        d[, v_next := self$V$predict(dt_next)]
        
        d[, target := get(self$reward_col) +
            self$gamma * get(self$mask_col) * v_next]
        
        self$Q$fit(d, y_col = "target", x_terms = self$q_terms)
      }
      invisible(self)
    },
    
    predict_Q = function(dt) self$Q$predict(dt),
    predict_V = function(dt) self$V$predict(dt)
  )
)
