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
    },
    
    fit = function(dt, n_iter = 30) {
      d <- as.data.table(copy(dt))
      if (!(self$mask_col %in% names(d))) d[, (self$mask_col) := 1.0]
      
      # ---- BOOTSTRAP Q_h ----
      d[, target0 := get(self$viol_col)]
      self$Q$fit(d, y_col = "target0", x_terms = self$q_terms)
      
      cat("\n SafetyLearner Start fitting\n")
      
      
      for (k in seq_len(n_iter)) {
        
        # ---- V_h update ----
        d[, qh_val := self$Q$predict(.SD),
          .SDcols = c("x", "a","hour", "xi_tilde")]
        self$V$fit(d, y_col = "qh_val", x_terms = self$v_terms)
        
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
      invisible(self)
    },
    
    predict_Q = function(dt) self$Q$predict(dt),
    predict_V = function(dt) self$V$predict(dt)
  )
)
