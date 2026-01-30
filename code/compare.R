rm(list = ls())
library(loo)
library(dplyr)

# Updated to work with compact saved format (no full fit object)
# Expected objects in RData: loo_result, bridge, plot_sum, fit_sum, posteriors, stan_data

resultsdir <- file.path("results")
datadir <- file.path("data")

# Load results from all four models
models <- c("bertrand", "2nd", "cournot", "moncom")
model_labels <- c("Bertrand", "Auction (2nd)", "Cournot", "Monopolistic Competition")
n_models <- length(models)

posteriors_list <- vector("list", n_models)
fits_sum <- vector("list", n_models)
bridges <- vector("list", n_models)
loo_list <- vector("list", n_models)

for (m in 1:n_models) {
  # Try new location first (results/), fallback to old (data/)
  infile <- file.path(resultsdir, paste0("stan_hhiperform_", models[m], ".RData"))
  if (!file.exists(infile)) {
    infile <- file.path(datadir, paste0("stan_hhiperform_", models[m], ".RData"))
  }

  if (!file.exists(infile)) {
    cat("Warning: File not found:", infile, "\n")
    next
  }

  cat("Loading", models[m], "results...\n")
  load(infile)

  # Check format: new compact format has 'posteriors' and 'loo_result',
  # old format has 'fit'
  if (exists("posteriors")) {
    # New compact format
    posteriors_list[[m]] <- posteriors
    loo_list[[m]] <- loo_result
    cat("  (compact format)\n")
  } else if (exists("fit")) {
    # Legacy format with full fit object - extract what we need
    cat("  (legacy format - extracting posteriors)\n")
    posteriors_list[[m]] <- list(
      a_event = rstan::extract(fit, "a_event")[[1]],
      s0 = rstan::extract(fit, "s0")[[1]],
      b_event = rstan::extract(fit, "b_event")[[1]],
      sigma_logshare = rstan::extract(fit, "sigma_logshare")[[1]],
      sigma_margin = rstan::extract(fit, "sigma_margin")[[1]],
      rho = rstan::extract(fit, "rho_gen")[[1]],
      gamma_loan = rstan::extract(fit, "gamma_loan")[[1]]
    )
    # Compute LOO from legacy fit
    cat("  Computing LOO for", models[m], "...\n")
    log_lik_m <- loo::extract_log_lik(fit)
    loo_list[[m]] <- loo::loo(log_lik_m, cores = 4)
  }

  fits_sum[[m]] <- fit_sum
  bridges[[m]] <- bridge

  # Clean up to avoid cross-contamination
  rm(list = intersect(c("fit", "posteriors", "loo_result", "bridge", "fit_sum", "plot_sum", "stan_data"), ls()))
}

# 1. Compute overall model probabilities using bridge sampling
cat("\n=== Computing Model Probabilities ===\n")

log_marg_likelihoods <- sapply(bridges, function(b) {
  if (is.null(b)) {
    return(NA)
  }
  b$logml
})

# Check for missing models
valid_models <- !is.na(log_marg_likelihoods)
if (sum(valid_models) < n_models) {
  cat("Warning: Some models missing. Using only:", models[valid_models], "\n")
}

# Compute posterior probabilities
prior_probs <- rep(1 / sum(valid_models), sum(valid_models))
log_prior <- log(prior_probs)
log_post_unnorm <- log_prior + log_marg_likelihoods[valid_models]
post_probs_valid <- exp(log_post_unnorm - max(log_post_unnorm))
post_probs_valid <- post_probs_valid / sum(post_probs_valid)

# Create full post_probs vector
post_probs <- rep(NA, n_models)
post_probs[valid_models] <- post_probs_valid

model_probs_df <- data.frame(
  Model = model_labels,
  Probability = post_probs,
  LogML = log_marg_likelihoods
)

print("Model Probabilities:")
print(model_probs_df)

# 2. LOO comparison (already computed, just compare)
cat("\n=== LOO Model Comparison ===\n")

# Filter to valid models with LOO results
valid_loo <- sapply(loo_list, function(x) !is.null(x) && !inherits(x, "try-error"))
if (sum(valid_loo) >= 2) {
  loo_compare_result <- loo_compare(loo_list[valid_loo])
  print("LOO Model Comparison:")
  print(loo_compare_result)
} else {
  cat("Not enough valid LOO results to compare.\n")
  loo_compare_result <- NULL
}

# 3. Bayesian Model Averaging (BMA) for a_event and s0
cat("\n=== Computing Bayesian Model Averaging ===\n")

# Extract parameter samples from posteriors
a_event_samples <- lapply(posteriors_list[valid_models], function(p) p$a_event)
s0_samples <- lapply(posteriors_list[valid_models], function(p) p$s0)

# Check dimensions
n_events <- dim(s0_samples[[1]])[2]
n_samples <- dim(s0_samples[[1]])[1]

cat("N events:", n_events, "\n")
cat("N posterior samples:", n_samples, "\n")

# Empty matrices for BMA draws
a_event_bma <- matrix(0, n_samples, n_events)
s0_bma <- matrix(0, n_samples, n_events)

# Compute weighted average across models
for (e in 1:n_events) {
  for (m_idx in 1:sum(valid_models)) {
    m <- which(valid_models)[m_idx]
    a_event_bma[, e] <- a_event_bma[, e] +
      post_probs_valid[m_idx] * a_event_samples[[m_idx]][, e]

    s0_bma[, e] <- s0_bma[, e] +
      post_probs_valid[m_idx] * s0_samples[[m_idx]][, e]
  }
}

# Summarize BMA results
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

# Convert s0 from logit scale to probability scale for interpretation
s0_bma_summary_prob <- s0_bma_summary
s0_bma_summary_prob[, c("mean", "q2.5", "q97.5")] <-
  plogis(s0_bma_summary[, c("mean", "q2.5", "q97.5")])

cat("\nBayesian Model Averaged Price Coefficients (first 10 events):\n")
print(head(a_event_bma_summary, 10))

cat("\nBayesian Model Averaged Outside Shares (first 10 events):\n")
print(head(s0_bma_summary_prob, 10))

# Save results
outdir <- if (dir.exists(resultsdir)) resultsdir else datadir
save(
  model_probs_df, loo_list, loo_compare_result,
  a_event_bma_summary, s0_bma_summary, s0_bma_summary_prob,
  file = file.path(outdir, "model_comparison_results.RData")
)

cat("\n=== Results saved to", file.path(outdir, "model_comparison_results.RData"), "===\n")
