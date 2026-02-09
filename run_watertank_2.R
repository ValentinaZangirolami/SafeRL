# ============================================================
# Offline safe RL with:
#  - belief estimation (demand) on demand_data
#  - reward critics (Q_r, V_r) on data_1
#  - safety critics (Q_h, V_h) on data_2
#  - weighted MLE policy on data_3
# ============================================================

#augmented dataset

suppressPackageStartupMessages({
  library(mgcv)
  library(dplyr)
  library(qgam)
  library(ggplot2)
  library(data.table)
})

# Load modules
source("R/models.R")
source("R/safe_rl.R")
source("R/policy.R")
source("R/safe_exoagent.R")
source("R/utils.R")

set.seed(42)
H.min = 5
H.min = 50
n_sample = 17600

load("generated_data.RData")

X<- make_dataset(knowledge, aug =TRUE)

# replace 0.3

X <- data_mix_policy(X, knowledge, replace_frac = 0.3)

day_random <- X$replaced_days
X <- X$X
names_to_remove <- c("hour_next", "synthetic", names(X)[1])
X[, (names_to_remove) := NULL]

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

#save(demand_data, reward_data, safety_data, policy_data,  file="aug_split_data.RData")

rm(knowledge, X)

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

eval_learning_aug <- agent$safety$log

# data test 

load("test_data.RData")

X_test <- make_dataset(knowledge)

names_to_keep<- c("hour", "h", "past_demand", "r")
X_test <- X_test[, ..names_to_keep]

#one-step rollout

X_pi <- apply_policy_one_step(dt = X_test, agent = agent)

#rollout replay 

traj_pi <- rollout_policy_replay(agent = agent, knowledge = knowledge)


# save data for evaluation
save(eval_learning_aug, traj_pi, X_pi, file="aug_data_eval.RData")