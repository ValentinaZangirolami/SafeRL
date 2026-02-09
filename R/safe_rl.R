# Estimation Q_h and V_h

library(R6)
library(data.table)


SafetyLearner <- R6Class(
  "SafetyLearner",
  public = list(
    gamma = NULL,
    critic_type = NULL,
    Q = NULL,
    V = NULL,
    q_terms = NULL,
    v_terms = NULL,
    viol_col = NULL,
    mask_col = NULL,
    
    probe_grid = NULL,  
    log = NULL,         
    
    initialize = function(gamma, critic_type,
                          Q_model, V_model,
                          q_terms, v_terms,
                          viol_col = "violation",
                          mask_col = "mask") {
      self$gamma <- gamma
      self$critic_type <- critic_type
      self$Q <- Q_model
      self$V <- V_model
      self$q_terms <- q_terms
      self$v_terms <- v_terms
      self$viol_col <- viol_col
      self$mask_col <- mask_col
      self$log <- data.table::data.table()
    },
    
    set_probe_grid = function(G) {
      G <- as.data.table(G)
      stopifnot(all(c("x","hour","xi_tilde") %in% names(G)))
      self$probe_grid <- G
      invisible(self)
    },
    
    .eval_feasibility = function(k, thr = 0) {
      stopifnot(!is.null(self$probe_grid))
      v <- self$V$predict(self$probe_grid)
      feasible <- v <= thr
      
      data.table::data.table(
        iter = as.integer(k),
        feasible_frac = mean(feasible, na.rm = TRUE),
        v_mean = mean(v, na.rm = TRUE),
        v_p95 = as.numeric(stats::quantile(v, 0.95, na.rm = TRUE)),
        v_max = max(v, na.rm = TRUE)
      )
    },
    
    fit = function(dt,
                   n_iter = 30,
                   log_every = 5,
                   thr = 0,
                   save_path = NULL,
                   save_rds = FALSE) {
      
      d <- as.data.table(copy(dt))
      if (!(self$mask_col %in% names(d))) d[, (self$mask_col) := 1.0]
      
      if (is.null(self$probe_grid)) {
        self$probe_grid <- make_probe_grid(
          x_min  = min(d$x, na.rm = TRUE),
          x_max  = max(d$x, na.rm = TRUE),
          xi_min = min(d$xi_tilde, na.rm = TRUE),
          xi_max = max(d$xi_tilde, na.rm = TRUE),
          hour_levels = sort(unique(d$hour)),
          nx = 50, nxi = 50
        )
      }
      
      
      # ---- BOOTSTRAP Q_h ----
      d[, target0 := get(self$viol_col)]
      self$Q$fit(d, y_col = "target0", x_terms = self$q_terms)
      
      
      for (k in seq_len(n_iter)) {
        
        # ---- V_h update ----
        d[, qh_val := self$Q$predict(.SD),
          .SDcols = c("x", "a", "hour", "xi_tilde")]
        self$V$fit(d, y_col = "qh_val", x_terms = self$v_terms)
        
        # ---- evaluation during learning ----
        if (k %% log_every == 0 || k == 1) {
          row <- self$.eval_feasibility(k = k, thr = thr)
          self$log <- data.table::rbindlist(list(self$log, row), use.names = TRUE, fill = TRUE)
          
          cat(sprintf(
            "  iter=%d  FeasibleFrac=%.3f  V_mean=%.3f  V_p95=%.3f  V_max=%.3f\n",
            row$iter, row$feasible_frac, row$v_mean, row$v_p95, row$v_max
          ))
        }
        
        # ---- Q_h update ----
        dt_next <- data.table(
          x = d[["x_next"]],
          hour = d[["hour_next"]],
          xi_tilde = d[["xi_tilde_next"]]
        )
        d[, vh_next := self$V$predict(dt_next)]
        
        if (self$critic_type == "hj") {
          d[, nonterminal :=
              (1 - self$gamma) * get(self$viol_col) +
              self$gamma * pmax(get(self$viol_col), vh_next)]
          d[, target :=
              nonterminal * get(self$mask_col) +
              get(self$viol_col) * (1 - get(self$mask_col))]
        } else {
          d[, target :=
              get(self$viol_col) +
              self$gamma * get(self$mask_col) * vh_next]
        }
        
        self$Q$fit(d, y_col = "target", x_terms = self$q_terms)
      }
      
      # save log 
      if (!is.null(save_path)) {
        if (isTRUE(save_rds)) {
          saveRDS(self$log, file = save_path)
        } else {
          data.table::fwrite(self$log, file = save_path)
        }
      }
      
      invisible(self)
    },
    
    predict_Q = function(dt) self$Q$predict(dt),
    predict_V = function(dt) self$V$predict(dt)
  )
)
