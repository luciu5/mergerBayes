rm(list=ls())
library(loo)
library(dplyr)
library(rstan)

datadir <- file.path("data")

# Load results from all four models
models <- c("bertrand", "2nd", "cournot", "moncom")
fits <- vector("list", 4)
fits_sum <- vector("list", 4)
bridges <- vector("list", 4)

for (m in 1:4) {
  # Each RData should contain a list named stan_results or similar
  obj <- load(file.path(datadir, paste0("stan_supreme_", models[m], ".RData")))
  
  # The loaded RData is assumed to create an object named "stan_results"
  # If your RData instead creates objects fit, fit_sum, bridge, adjust here
  if (exists("stan_results")) {
    fits[[m]]     <- stan_results$fit
    fits_sum[[m]] <- stan_results$fit_sum
    bridges[[m]]  <- stan_results$bridge
  } else {
    # fallback for old-style RData containing fit, fit_sum, bridge
    fits[[m]]     <- fit
    fits_sum[[m]] <- fit_sum
    bridges[[m]]  <- bridge
  }
}

# 1. Compute overall model probabilities
log_marg_likelihoods <- sapply(bridges, function(b) b$logml)
prior_probs <- rep(1/4, 4)
log_prior <- log(prior_probs)
log_post_unnorm <- log_prior + log_marg_likelihoods
post_probs <- exp(log_post_unnorm - max(log_post_unnorm))
post_probs <- post_probs / sum(post_probs)

model_probs_df <- data.frame(
  Model = models,
  Probability = post_probs,
  LogML = log_marg_likelihoods
)

print("Model Probabilities:")
print(model_probs_df)

# 2. Compute market/time-specific model probabilities via LOO
loo_list <- vector("list", 4)

for (m in 1:4) {
  log_lik_m <- extract_log_lik(fits[[m]])   # calls stan object correctly
  loo_list[[m]] <- loo(log_lik_m, cores=4)
}

# Function for per-market, per-year model probabilities
get_market_time_probs <- function(event_id_val, year_val) {
  idx <- which(simdata$event_id == event_id_val &
                 simdata$year == year_val)
  
  model_lls <- numeric(4)
  
  for (m in 1:4) {
    log_lik_m <- extract_log_lik(fits[[m]])
    model_lls[m] <- sum(colMeans(log_lik_m[, idx]))
  }
  
  lls_centered <- model_lls - max(model_lls)
  probs <- exp(lls_centered) / sum(exp(lls_centered))
  return(probs)
}

# 3. Bayesian Model Averaging (BMA) for a_event_rescaled and s0

a_event_samples <- lapply(fits, function(f) {
  rstan::extract(f, pars="a_event_rescaled")[[1]]
})

s0_samples <- lapply(fits, function(f) {
  rstan::extract(f, pars="s0")[[1]]
})

n_events <- dim(s0_samples[[1]])[2]
n_samples <- dim(s0_samples[[1]])[1]

# Empty matrices for BMA draws
a_event_bma <- matrix(0, n_samples, n_events)
s0_bma <- matrix(0, n_samples, n_events)

for (e in 1:n_events) {
  for (m in 1:4) {
    a_event_bma[,e] <- a_event_bma[,e] +
      post_probs[m] * a_event_samples[[m]][,e]
    
    s0_bma[,e] <- s0_bma[,e] +
      post_probs[m] * s0_samples[[m]][,e]
  }
}

a_event_bma_summary <- data.frame(
  event = 1:n_events,
  mean = colMeans(a_event_bma),
  sd = apply(a_event_bma, 2, sd),
  q2.5 = apply(a_event_bma, 2, quantile, 0.025),
  q97.5 = apply(a_event_bma, 2, quantile, 0.975)
)

s0_bma_summary <- data.frame(
  event = 1:n_events,
  mean = colMeans(s0_bma),
  sd = apply(s0_bma, 2, sd),
  q2.5 = apply(s0_bma, 2, quantile, 0.025),
  q97.5 = apply(s0_bma, 2, quantile, 0.975)
)

print("Bayesian Model Averaged Price Coefficients:")
print(a_event_bma_summary)

print("Bayesian Model Averaged Outside Shares:")
print(s0_bma_summary)



