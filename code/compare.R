rm(list = ls())
library(loo)
library(dplyr)
library(rstan)
library(bridgesampling)

# ------------------------------------------------------------------------------
# SETUP & CONFIGURATION
# ------------------------------------------------------------------------------
resultsdir <- file.path("results")
datadir <- file.path("data")

# Define Models
models <- c("bertrand", "2nd", "cournot", "moncom")
model_labels <- c("Bertrand", "Auction (2nd)", "Cournot", "Monopolistic Competition")
n_models <- length(models)

# Storage Lists
posteriors_list <- vector("list", n_models)
fits_sum <- vector("list", n_models)
bridges <- vector("list", n_models)
loo_list <- vector("list", n_models)

# ------------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------------
cat("=== Loading Model Results ===\n")

for (m in 1:n_models) {
  # Try results/ first, then data/
  infile <- file.path(resultsdir, paste0("stan_hhiperform_", models[m], ".RData"))
  if (!file.exists(infile)) {
    infile <- file.path(datadir, paste0("stan_hhiperform_", models[m], ".RData"))
  }

  if (!file.exists(infile)) {
    cat(sprintf("[MISSING] %s results not found.\n", models[m]))
    next
  }

  cat(sprintf("[LOADING] %s...", models[m]))
  load(infile)

  # HANDLE FORMAT: Check if new compact format or legacy fit object
  if (exists("posteriors")) {
    posteriors_list[[m]] <- posteriors
    if (exists("loo_result")) loo_list[[m]] <- loo_result
    cat(" (compact)\n")
    
  } else if (exists("fit")) {
    cat(" (legacy - extracting)\n")
    # Extract only what we need to save memory
    posteriors_list[[m]] <- list(
      a_event = rstan::extract(fit, "a_event")[[1]],
      s0 = rstan::extract(fit, "s0")[[1]], 
      # Add other parameters if needed for plots
      rho = tryCatch(rstan::extract(fit, "rho_gen")[[1]], error=function(e) NULL)
    )
    # Compute LOO on the fly if missing
    log_lik_m <- loo::extract_log_lik(fit)
    loo_list[[m]] <- loo::loo(log_lik_m, cores = 4)
  }
  
  # Store Bridge Result
  if (exists("bridge")) bridges[[m]] <- bridge
  if (exists("fit_sum")) fits_sum[[m]] <- fit_sum

  # Clean environment for next iteration
  rm(list = intersect(c("fit", "posteriors", "loo_result", "bridge", "fit_sum", "plot_sum", "stan_data"), ls()))
}

# ------------------------------------------------------------------------------
# 2. MODEL PROBABILITIES (BRIDGE SAMPLING)
# ------------------------------------------------------------------------------
cat("\n=== Computing Posterior Model Probabilities ===\n")

log_marg_likelihoods <- sapply(bridges, function(b) {
  if (is.null(b)) return(NA)
  # Handle numerical instability
  if (is.nan(b$logml) || is.infinite(b$logml)) return(-Inf)
  return(b$logml)
})

# Filter valid models
valid_models <- !is.na(log_marg_likelihoods) & !is.infinite(log_marg_likelihoods)
if (sum(valid_models) < 2) {
  warning("Fewer than 2 valid models found. BMA may be trivial.")
}

# Calculate Probabilities: P(M|D) propto P(D|M) * P(M)
# Using log-sum-exp trick for numerical stability
prior_probs <- rep(1 / sum(valid_models), sum(valid_models))
log_prior <- log(prior_probs)
log_ml_valid <- log_marg_likelihoods[valid_models]

log_post_unnorm <- log_prior + log_ml_valid
# Shift by max to prevent overflow
post_probs_valid <- exp(log_post_unnorm - max(log_post_unnorm))
post_probs_valid <- post_probs_valid / sum(post_probs_valid)

# Create Result Vector
post_probs <- rep(NA, n_models)
post_probs[valid_models] <- post_probs_valid

# Display Table
model_probs_df <- data.frame(
  Model = model_labels,
  Log_ML = round(log_marg_likelihoods, 2),
  Probability = round(post_probs, 4)
)
print(model_probs_df)

# ------------------------------------------------------------------------------
# 3. BAYESIAN MODEL AVERAGING (MIXTURE SAMPLING)
# ------------------------------------------------------------------------------
cat("\n=== Computing BMA Distributions ===\n")

if (sum(valid_models) > 0) {
  valid_indices <- which(valid_models)
  
  # Extract samples from valid models
  # Structure: List of [N_draws x N_events]
  a_samples <- lapply(posteriors_list[valid_indices], function(p) p$a_event)
  s0_samples <- lapply(posteriors_list[valid_indices], function(p) p$s0)
  
  n_draws <- nrow(a_samples[[1]])
  n_events <- ncol(a_samples[[1]])
  
  # A. Generate Mixture Indices
  # Instead of averaging (0.5*A + 0.5*B), we draw samples from A or B
  # This preserves the true multimodal shape of the posterior.
  set.seed(2025)
  model_choices <- sample(
    x = 1:length(valid_indices),
    size = n_draws,
    replace = TRUE,
    prob = post_probs_valid
  )
  
  # B. Build BMA Matrix
  a_bma <- matrix(NA, n_draws, n_events)
  s0_bma <- matrix(NA, n_draws, n_events)
  
  for (i in 1:n_draws) {
    m_idx <- model_choices[i]
    a_bma[i, ] <- a_samples[[m_idx]][i, ]
    s0_bma[i, ] <- s0_samples[[m_idx]][i, ]
  }
  
  # C. Summarize Results
  get_summary <- function(mat) {
    data.frame(
      Event_ID = 1:ncol(mat),
      Mean = colMeans(mat),
      SD = apply(mat, 2, sd),
      Q2.5 = apply(mat, 2, quantile, 0.025),
      Q50 = apply(mat, 2, median),
      Q97.5 = apply(mat, 2, quantile, 0.975)
    )
  }
  
  a_summary <- get_summary(a_bma)
  
  # Transform s0 to probability scale BEFORE summarizing
  # (Standard Logit s0 is often on logit scale in Stan, check your stan code)
  # Assuming s0 in posteriors is already inverse-logit (0-1), usually true for generated quantities.
  # If s0 is logit, wrap in plogis(). Let's assume it's already 0-1 based on your previous code.
  s0_summary <- get_summary(s0_bma) 
  
  cat("\nTop 5 Events (Price Sensitivity BMA):\n")
  print(head(a_summary, 5))
  
  cat("\nTop 5 Events (Outside Share BMA):\n")
  print(head(s0_summary, 5))
  
  # D. Save
  outfile <- file.path(if (dir.exists(resultsdir)) resultsdir else datadir, "bma_results.RData")
  save(model_probs_df, a_summary, s0_summary, a_bma, s0_bma, loo_list, file = outfile)
  cat(sprintf("\n[SUCCESS] Results saved to %s\n", outfile))
  
} else {
  cat("\n[ERROR] No valid models found. Skipping BMA.\n")
}

# ------------------------------------------------------------------------------
# 4. LOO COMPARISON (Predictive Accuracy)
# ------------------------------------------------------------------------------
cat("\n=== LOO Comparison ===\n")
valid_loo <- !sapply(loo_list, is.null)
if (sum(valid_loo) >= 2) {
  print(loo_compare(loo_list[valid_loo]))
} else {
  cat("Insufficient LOO results for comparison.\n")
}
