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
      
      if (!("mask" %in% names(dt))) dt[, mask := 1.0]
      dt[, hour := as.integer(hour)]
      if (!("hour_next" %in% names(dt))) {
        dt[, hour_next := ifelse(hour == 24L, 1L, as.integer(hour + 1L))]
      }
      
      dt[, .(
        x = h,
        a = a,
        hour = hour,
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
      
      dt <- dt[!is.na(past_demand) & !is.na(current_demand)]
      if (!("mask" %in% names(dt))) dt[, mask := 1.0]
      dt[, hour := as.integer(hour)]
      
      b0 <- self$belief$predict(dt[, .(hour, h, past_demand)])
      dt[, est_demand := b0$est]
      dt[, est_max_demand := b0$est_max]
      
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
                   cost_ub = 200.0,
                   log_every = 5,
                   safety_log_path = "results/safety_feasible_log.csv") {
      
      self$gamma <- gamma
      
      cat("\n[Agent] Fitting belief model...\n")
      self$belief <- BeliefModel$new(tau = belief_tau)
      self$belief$fit(as.data.table(demand_data))
      cat("[Agent] Belief model fitted.\n")
      
      # augment with belief
      reward_aug <- self$.augment_with_belief(reward_data)
      safety_aug <- self$.augment_with_belief(safety_data)
      policy_aug <- self$.augment_with_belief(policy_data)
      
      # standardize
      dt_reward <- self$.standardize(reward_aug)
      dt_safety <- self$.standardize(safety_aug)
      dt_policy <- self$.standardize(policy_aug)
      
      # --- Reward learner  ---
      self$reward <- RewardLearner$new(
        gamma = gamma,
        Q_model = GAMRegressor$new(),
        V_model = ExpectileGAM$new(tau = reward_tau, n_irls = 8),
        
        
        q_terms = c("s(x, k=10)", "s(hour, bs='cc', k=8)", "s(a, k=10)"),
        v_terms = c("s(x, k=10)", "s(hour, bs='cc', k=8)"),
        reward_col = "r",
        mask_col = "mask"
      )
      cat("\n[Agent] Fitting reward critics...\n")
      self$reward$fit(dt_reward, n_iter = 30)
      
      # --- Safety learner ---
      self$safety <- SafetyLearner$new(
        gamma = gamma,
        critic_type = critic_type,
        Q_model = GAMRegressor$new(),
        V_model = ExpectileGAM$new(tau = 1 - safety_tau, n_irls = 8),
        q_terms = c("s(x, k=10)", "s(hour, bs='cc', k=8)", "s(a, k=10)", "s(xi_tilde, k=10)"),
        v_terms = c("s(x, k=10)", "s(hour, bs='cc', k=8)", "s(xi_tilde, k=10)"),
        viol_col = "violation",
        mask_col = "mask"
      )
      
      # eval
      G <- make_probe_grid(
        x_min  = min(dt_safety$x, na.rm = TRUE),
        x_max  = max(dt_safety$x, na.rm = TRUE),
        xi_min = min(dt_safety$xi_tilde, na.rm = TRUE),
        xi_max = max(dt_safety$xi_tilde, na.rm = TRUE),
        hour_levels = 1:24,
        nx = 50, nxi = 50
      )
      if ("set_probe_grid" %in% names(self$safety)) self$safety$set_probe_grid(G)
      
      cat("\n[Agent] Fitting safety critics...\n")
      
      try({
        self$safety$fit(
          dt_safety,
          n_iter = 30,
          log_every = log_every,
          save_path = safety_log_path
        )
      }, silent = TRUE)
      
    
      if (is.null(self$safety$log) || nrow(self$safety$log) == 0) {
        self$safety$fit(dt_safety, n_iter = 30)
      }
      
      # --- Policy ---
      self$weighter <- FeasibilityWeighter$new(
        cost_temperature = cost_temperature,
        reward_temperature = reward_temperature,
        cost_ub = cost_ub
      )
      dp <- self$weighter$compute(dt_policy, self$reward, self$safety)
      
      # bounded policy 
      self$policy <- WeightedGaussianPolicy$new(
        mu_terms = c("s(x, k=10)", "s(hour, bs='cc', k=8)"),
        a_min = 0,
        a_max = 10,
        deterministic = TRUE
      )
      cat("\n[Agent] Fitting policy...\n")
      self$policy$fit(dp, w_col = "w", a_col = "a")
      
      invisible(self)
    },
    
    act = function(hour, h, past_demand) {
      hour <- as.integer(hour)
      
      b <- self$belief$predict(data.table(hour = hour, h = h, past_demand = past_demand))
      xi_tilde <- b$est_max
      
      a <- self$policy$act(data.table(x = h, hour = hour))
      list(a = as.numeric(a), xi_tilde = as.numeric(xi_tilde))
    }
  )
)
