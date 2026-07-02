# SafeRL

(ongoing project)

Offline safe reinforcement learning for water-tank control with exogenous demand uncertainty.

This repository contains an R implementation of an offline safe RL pipeline for a water-tank system. The goal is to learn a policy that minimizes pumping costs while keeping the tank level within safety limits under uncertain water demand.

The implementation follows an Exogenous State Markov Decision Process setting, where the system state includes the endogenous tank level and an exogenous demand process. 

The data generating mechanism has been taken from [this repository](https://github.com/acandelieri/SafeExploringControlPolicies), related to [1].

## Repository structure

```text
.
├── DataGenerator.R          # Water-tank data generator
├── GeneratingData.R         # Script to generate training and test data
├── generated_data.RData     # Pre-generated training data
├── test_data.RData          # Pre-generated test data
├── run_watertank.R          # Main offline safe RL pipeline
├── run_watertank_2.R        # Augmented-data version of the pipeline
├── saferl.Rproj             # RStudio project file
├── README.md
└── R/
    ├── models.R             # GAM, expectile GAM, belief quantile model, policy and reward learner
    ├── policy.R             # Feasibility-based policy weighting
    ├── safe_exoagent.R      # Main safe exogenous RL agent
    ├── safe_rl.R            # Safety critic learning
    └── utils.R              # Dataset construction and evaluation utilities
```

### Generate data

To regenerate the synthetic training and test datasets, run:

```r
source("GeneratingData.R")
```

This creates:

```text
generated_data.RData
test_data.RData
```

The generated trajectories contain 24 hourly decision steps per day.

### Run the standard offline safe RL pipeline

```r
source("run_watertank.R")
```

### Run the augmented-data pipeline

```r
source("run_watertank_2.R")
```


## Dependencies

The implementation uses the following R packages:

```r
install.packages(c(
  "R6",
  "data.table",
  "mgcv",
  "qgam",
  "dplyr",
  "ggplot2",
  "plot3D",
  "DiceKriging"
))
```

### References

[1] Candelieri, A. et al. (2023). *Safe Optimal Control of Dynamic Systems: Learning from Experts and Safely Exploring New Policies*. Mathematics 11(20).

## Citation

If you use this code, please cite:

Zangirolami, V., Pavesi, F. and Zanotti, M. (2026). *Safe Exogenous State Reinforcement Learning for water tank system*. Statistical Science: From Theory to Applied Research IV (to appear).

```bibtex
@incollection{zangirolami2026safeexo,
  author    = {Zangirolami, Valentina and Pavesi, Federico and Zanotti, Marco},
  title     = {Safe Exogenous State Reinforcement Learning for Water Tank System},
  booktitle = {Statistical Science: From Theory to Applied Research IV},
  year      = {2026}
}
```


