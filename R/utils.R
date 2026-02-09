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

# eval test

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

get_reward <- function(hour, action){
  prices = c(rep(1,6),rep(3,2),rep(2,10),rep(3,2),rep(2,2),rep(1,2))
  reward = prices[hour] * (1/eta) * action^(1/3)
}
