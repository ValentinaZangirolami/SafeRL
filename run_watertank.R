# ============================================================
# Offline safe RL with:
#  - demand estimation on demand_data
#  - reward critics (Q_r, V_r) on data_1
#  - safety critics (Q_h, V_h) on data_2
#  - weighted MLE policy on data_3
# ============================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(dplyr)
  library(qgam)
  library(data.table)
})

set.seed(42)

max.inflow = 10
H.min = 5
H.max = 50
prices = c(rep(1,6),rep(3,2),rep(2,10),rep(3,2),rep(2,2),rep(1,2))
eta = 0.85
n_sample = 17600

load("generated_data.RData")

N = nrow(knowledge$s) * ncol(knowledge$s)
X = data.frame(
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
    X$hour[k]      = j
    X$h[k]      = round(knowledge$s[i,j], 4)
    X$a[k]      = round(knowledge$a[i,j], 4)
    X$h_next[k] = round(knowledge$s_[i,j], 4)
    X$r[k]      = round(knowledge$r[i,j], 4)
    demand_true <- X$h[k] + X$a[k] - X$h_next[k] 
    X$violation[k] = demand_true - X$h[k]
    
    if (j != 1) {
      X$current_demand[k] = round(abs(X$h[k] + X$a[k] - X$h_next[k]), 4)
      X$past_demand[k]    = round(abs(X$h[k-1] + X$a[k-1] - X$h_next[k-1]), 4)
    } else {
      X$current_demand[k] = NA_real_
      X$past_demand[k]    = NA_real_
    }
    k = k + 1
  }
}

# Split 
set.seed(42)
grp <- integer(N)
grp[sample.int(N, 17600)] <- 1
rest <- which(grp == 0)
grp[rest] <- sample(rep(2:4, length.out = length(rest)))

demand_data <- X[grp == 1, ]
reward_data <- X[grp == 2, ]
safety_data <- X[grp == 3, ]
policy_data <- X[grp == 4, ]

rm(knowledge, X)

# Load modules
source("R/models.R")
source("R/safe_rl.R")
source("R/policy.R")
source("R/safe_exoagent.R")

# Fit agent
agent <- SafeExoAgent$new()
agent$fit(
  demand_data = demand_data,
  reward_data = reward_data,
  safety_data = safety_data,
  policy_data = policy_data,
  belief_tau = 0.95,
  gamma = 0.99,
  reward_tau = 0.8,
  safety_tau = 0.8,
  critic_type = "hj",
  cost_temperature = 3.0,
  reward_temperature = 3.0,
  cost_ub = 200.0
)

print(agent$act(hour = 10, h = 20, past_demand = 5))
