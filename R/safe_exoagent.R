# Agent

library(R6)
library(data.table)


SafeExoAgent <- R6Class(
  "SafeExoAgent",
  public = list(
    gamma = NULL,
    
    belief = NULL,
    reward = NULL,
    safety = NULL,
    weighter = NULL,
    policy = NULL,
    
  
    .standardize = function(dt_aug) {
      dt <- as.data.table(copy(dt_aug))
      dt[, .(
        x = h,
        a = a,
        hour=hour,
        x_next = h_next,
        hour_next = hour_next,
        r = r,
        xi_tilde = est_max_demand,
        xi_tilde_next = est_max_demand_next,
        violation = violation,
        mask = mask
      )]
    },
    
    .augment_with_belief = function(dt_raw) {
      dt <- as.data.table(copy(dt_raw))
      
      dt <- dt[!is.na(past_demand)]
      if (!("mask" %in% names(dt))) dt[, mask := 1.0]
      
      # current demand
      b0 <- self$belief$predict(dt[, .(hour, h, past_demand)])
      dt[, est_demand := b0$est]
      dt[, est_max_demand := b0$est_max]
      
      # next demand
      dt[, hour_next := ifelse(hour == 24L, 1L, as.integer(hour + 1L))]
      dt_next <- dt[, .(hour = hour_next, h = h_next, past_demand = current_demand)]
      b1 <- self$belief$predict(dt_next)
      dt[, est_max_demand_next := b1$est_max]
      
      dt
    },
    
    
    fit = function(demand_data,
                   reward_data,
                   safety_data,
                   policy_data,
                   belief_tau = 0.95,
                   gamma = 0.99,
                   reward_tau = 0.8,
                   safety_tau = 0.8,
                   critic_type = "hj",
                   cost_temperature = 3.0,
                   reward_temperature = 3.0,
                   cost_ub = 200.0) {
      
  
      self$gamma <- gamma
      
      cat("\n[1] Fitting belief model...\n")
      
      # exo estimation
      self$belief <- BeliefModel$new(tau = belief_tau)
      self$belief$fit(as.data.table(demand_data))
      
      cat("End Belief model fitted\n")
      
      reward_aug <- self$.augment_with_belief(reward_data)
      safety_aug <- self$.augment_with_belief(safety_data)
      policy_aug <- self$.augment_with_belief(policy_data)
      
      dt_reward <- self$.standardize(reward_aug)
      dt_safety <- self$.standardize(safety_aug)
      dt_policy <- self$.standardize(policy_aug)
      
      # value reward
      self$reward <- RewardLearner$new(
        gamma = gamma,
        Q_model = GAMRegressor$new(),
        V_model = ExpectileGAM$new(tau = reward_tau, n_irls = 8),
        q_terms = c("s(x, k = 1)", "s(hour, bs='cc', k=3)", "s(a, k = 1)"),
        v_terms = c("s(x, k = 1)", "s(hour, bs='cc', k=3)"),
        reward_col = "r",
        mask_col = "mask"
      )
      self$reward$fit(dt_reward, n_iter = 30)
      
      # safe value
      self$safety <- SafetyLearner$new(
        gamma = gamma,
        critic_type = critic_type,
        Q_model = GAMRegressor$new(),
        V_model = ExpectileGAM$new(tau = 1 - safety_tau, n_irls = 8),
        q_terms = c("s(x, k=1)", "s(hour, bs='cc', k=3)", "s(a, k=1)", "s(xi_tilde, k=3)"),
        v_terms = c("s(x, k=1)", "s(hour, bs='cc', k=3)", "s(xi_tilde, k=3)"),
        viol_col = "violation",
        mask_col = "mask"
      )
      self$safety$fit(dt_safety, n_iter = 30)
      
      # policy estimation
      self$weighter <- FeasibilityWeighter$new(
        cost_temperature = cost_temperature,
        reward_temperature = reward_temperature,
        cost_ub = cost_ub
      )
      dp <- self$weighter$compute(dt_policy, self$reward, self$safety)
      
      self$policy <- WeightedGaussianPolicy$new(mu_terms = c("s(x,  k=1)", "s(hour, bs='cc', k=3)"), deterministic = TRUE)
      self$policy$fit(dp, w_col = "w", a_col = "a")
      
      invisible(self)
    },
    
    act = function(hour, h, past_demand) {
      # estimated exogenous v.
      b <- self$belief$predict(data.table(hour = hour, h = h, past_demand = past_demand))
      xi_tilde <- b$est_max
      
      #actions
      a <- self$policy$act(data.table(x = h, hour = hour))
      list(a = as.numeric(a), xi_tilde = as.numeric(xi_tilde))
    }
  )
)
