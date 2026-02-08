# policy estimation

library(R6)
library(data.table)

# weights
FeasibilityWeighter <- R6Class(
  "FeasibilityWeighter",
  public = list(
    cost_temperature = NULL,
    reward_temperature = NULL,
    cost_ub = NULL,
    
    initialize = function(cost_temperature = 3.0, reward_temperature = 3.0, cost_ub = 200.0) {
      self$cost_temperature <- cost_temperature
      self$reward_temperature <- reward_temperature
      self$cost_ub <- cost_ub
    },
    
    compute = function(dt, reward_learner, safety_learner) {
      d <- as.data.table(copy(dt))
      
      cat("\n Policy Start fitting\n")
      
      d[, q := reward_learner$predict_Q(.SD), .SDcols = c("x", "a", "hour")]
      d[, v := reward_learner$predict_V(.SD), .SDcols = c("x", "hour")]
      
      d[, qc := safety_learner$predict_Q(.SD), .SDcols = c("x", "a", "hour", "xi_tilde")]
      d[, vc := safety_learner$predict_V(.SD), .SDcols = c("x", "hour", "xi_tilde")]
      
      unsafe_condition <- as.numeric(d[["vc"]] > 0)
      safe_condition   <- as.numeric(d[["vc"]] <= 0) * as.numeric(d[["qc"]] <= 0)
      
      cost_exp_adv   <- exp((d[["vc"]] - d[["qc"]]) * self$cost_temperature)
      reward_exp_adv <- exp((d[["q"]]  - d[["v"]])  * self$reward_temperature)
      
      unsafe_w <- unsafe_condition * pmin(cost_exp_adv, self$cost_ub)
      safe_w   <- safe_condition   * pmin(reward_exp_adv, 100)
      
      d[, w := unsafe_w + safe_w]
      d[!is.finite(w) | w < 0, w := 0]
      
      d
    }
  )
)
