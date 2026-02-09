# functions

make_probe_grid <- function(x_min, x_max, xi_min, xi_max,
                            hour_levels = 1:24,
                            nx = 50, nxi = 50) {
  data.table::CJ(
    x = seq(x_min, x_max, length.out = nx),
    xi_tilde = seq(xi_min, xi_max, length.out = nxi),
    hour = as.integer(hour_levels)
  )
}

data_mix_policy <- function(X, knowledge,
                                            replace_frac = 0.1,
                                            max_inflow = 10,
                                            H_min = 5,
                                            eta = 0.85,
                                            seed = 123,
                                            clip_upper = FALSE,
                                            H_max = 50) {
  set.seed(seed)
  X <- data.table::as.data.table(data.table::copy(X))
  stopifnot(all(c("day","hour","h","h_next","a","r","current_demand","past_demand","violation") %in% names(X)))
  
  # choose days to replace
  days <- sort(unique(X$day))
  n_days <- length(days)
  m <- max(1, floor(replace_frac * n_days))
  days_rep <- sample(days, m, replace = FALSE)
  
  # true demand per day/hour from original data
  S  <- knowledge$s
  A  <- knowledge$a
  S_ <- knowledge$s_
  D_true <- abs(S + A - S_)  # true demand
  
  syn_list <- vector("list", length(days_rep))
  
  for (idx in seq_along(days_rep)) {
    d_id <- days_rep[idx]
    d_seq <- as.numeric(D_true[d_id, ])
    t_max <- length(d_seq)
    
    h <- as.numeric(S[d_id, 1])
    
    hour <- 1:t_max
    hour_next <- ifelse(hour == 24L, 1L, hour + 1L)
    
    h_vec <- numeric(t_max)
    a_vec <- numeric(t_max)
    h_next_vec <- numeric(t_max)
    r_vec <- numeric(t_max)
    viol_vec <- numeric(t_max)
    
    cur_dem <- abs(d_seq)
    past_dem <- c(NA, abs(d_seq[-T]))
    
    for (t in 1:t_max) {
      a <- runif(1, 0, max_inflow)
      
      h_next <- h + a - d_seq[t]
      
      if (isTRUE(clip_upper)) h_next <- pmin(h_next, H_max)
      
      violation <- d_seq[t] - (h - H_min)
      
      r <- get_reward(hour = t, action = a, eta = eta)
      
      h_vec[t] <- h
      a_vec[t] <- a
      h_next_vec[t] <- h_next
      viol_vec[t] <- violation
      r_vec[t] <- r
      
      h <- h_next
    }
    
    syn_list[[idx]] <- data.table::data.table(
      day = d_id,
      hour = hour,
      hour_next = as.integer(hour_next),
      h = round(h_vec, 4),
      a = round(a_vec, 4),
      h_next = round(h_next_vec, 4),
      r = round(r_vec, 6),
      current_demand = round(cur_dem, 4),
      past_demand = round(past_dem, 4),
      violation = round(viol_vec, 4),
      synthetic = 1L
    )
  }
  
  syn_dt <- data.table::rbindlist(syn_list)
  
  X[, synthetic := 0L]
  X_kept <- X[!day %in% days_rep]
  
  X_new <- data.table::rbindlist(list(X_kept, syn_dt), use.names = TRUE, fill = TRUE)
  stopifnot(nrow(X_new) == nrow(X))
  
  list(X = X_new[order(day, hour)], replaced_days = days_rep)
}

make_dataset = function(knowledge, H.min=5, aug = FALSE){
  N = nrow(knowledge$s) * ncol(knowledge$s)
  X = data.frame(
    day <- integer(N),
    hour = numeric(N),
    h = numeric(N),
    a = numeric(N),
    h_next = numeric(N),
    r = numeric(N),
    current_demand = numeric(N),
    past_demand = numeric(N),
    violation = numeric(N)
  )
  
  k = 1
  for (i in 1:nrow(knowledge$s)) {
    for (j in 1:ncol(knowledge$s)) {
      X$day[k] <- i
      X$hour[k]      = j
      X$h[k]      = round(knowledge$s[i,j], 4)
      X$a[k]      = round(knowledge$a[i,j], 4)
      X$h_next[k] = round(knowledge$s_[i,j], 4)
      X$r[k]      = - round(knowledge$r[i,j], 4)
      demand_true <- X$h[k] + X$a[k] - X$h_next[k] 
      X$violation[k] = demand_true - (X$h[k]- H.min)
      
      if (j != 1) {
        X$current_demand[k] = round(abs(X$h[k] + X$a[k] - X$h_next[k]), 4)
        X$past_demand[k]    = X$current_demand[k-1]
      } else {
        X$current_demand[k] = round(abs(X$h[k] + X$a[k] - X$h_next[k]), 4)
        X$past_demand[k]    = 0
      }
      k = k + 1
    }
  }
  X <- if (aug) X else X[, -1]
  return(X)
}

# eval test

test_policy_one_step <- function(dt, agent,
                                  H_min = 5,
                                  max_inflow = 10,
                                  eta = 0.85,
                                  only_hours = 1:23) {
  dt <- data.table::as.data.table(data.table::copy(dt))
  
  # hours from 1 to 23
  dt <- dt[hour %in% only_hours]
  dt <- dt[!is.na(past_demand)]
  
  # compute action from agent
  out <- dt[, {
    res <- agent$act(hour = hour, h = h, past_demand = past_demand)
    list(a_pi = as.numeric(res$a),
         xi_tilde = as.numeric(res$xi_tilde))
  }, by = .(day, hour)]
  
  dt <- merge(dt, out, by = c("day", "hour"), all.x = TRUE)
  
  dt[, a_pi := pmin(pmax(a_pi, 0), max_inflow)]
  
  dt[, h_next_pi := h + a_pi - current_demand]
  
  dt[, r_pi := get_reward(hour = hour, action = a_pi, eta = eta)]
  
  dt[, violation_pi := current_demand - (h - H_min)]
  
  dt[, hour_next := ifelse(hour == 24L, 1L, hour + 1L)]
  
  dt
}

rollout_policy_replay <- function(agent,
                                  knowledge,
                                  H_min = 5,
                                  eta = 0.85,
                                  max_inflow = 10) {
  
  D_true <- abs(knowledge$s + knowledge$a - knowledge$s_)  
  S0 <- knowledge$s[, 1]
  
  n_days <- nrow(D_true)
  t_max <- ncol(D_true)
  
  out <- vector("list", n_days)
  
  for (day in 1:n_days) {
    d_seq <- as.numeric(D_true[day, ])
    h <- as.numeric(S0[day])
    
    past_demand <- 0
    
    dt_day <- data.table::data.table(
      day = day,
      hour = 1:t_max,
      h = NA_real_,
      past_demand = NA_real_,
      demand_true = d_seq,   
      a_pi = NA_real_,
      h_next = NA_real_,
      r_pi = NA_real_,
      violation_pi = NA_real_,
      xi_tilde = NA_real_
    )
    
    dt_day[hour == 1, `:=`(
      h = h,
      past_demand = past_demand
    )]
    
    for (t in 1:t_max) {
      # decision happens at state h (which is the state at hour t-1)
      act <- agent$act(hour = t, h = h, past_demand = past_demand)
      a <- as.numeric(act$a)
      xi_tilde <- as.numeric(act$xi_tilde)
      
      a <- pmin(pmax(a, 0), max_inflow)
      
      r <- get_reward(hour = t, action = a, eta = eta)
      
      # demand for this transition
      d_t <- d_seq[t]
      
      viol <- d_t - (h - H_min)
      h_next <- h + a - d_t
      
      
      dt_day[hour == t, `:=`(h = ..h, past_demand = ..past_demand, h_next = ..h_next, xi_tilde = ..xi_tilde, a_pi = a, r_pi = r,
                             violation_pi = viol)]
      
      
      # advance
      h <- h_next
      past_demand <- abs(d_t)
    }
    
    out[[day]] <- dt_day
  }
  
  data.table::rbindlist(out)
}

#violations
# x_mat has to be N x T

count_state_violations <- function(x_mat, H_min = 5, H_max = 50) {
  stopifnot(is.matrix(x_mat) || is.data.frame(x_mat))
  x_mat <- as.matrix(x_mat)
  
  above <- x_mat > H_max
  below <- x_mat < H_min
  viol  <- above | below
  
  viol_max <- pmax(x_mat - H_max, 0) + pmax(H_min - x_mat, 0)
  
  data.table::data.table(
    day = seq_len(nrow(x_mat)),
    n_above_max = rowSums(above, na.rm = TRUE),
    n_below_min = rowSums(below, na.rm = TRUE),
    n_violations = rowSums(viol, na.rm = TRUE),
    violation_rate = rowMeans(viol, na.rm = TRUE),
    max_violation = apply(viol_max, 1, max, na.rm = TRUE)
  )
}

#costs 

get_reward <- function(hour, action, eta=0.85){
  prices = c(rep(1,6),rep(3,2),rep(2,10),rep(3,2),rep(2,2),rep(1,2))
  reward = -(prices[hour] * (1/eta) * action^(1/3))
  return(reward)
}
