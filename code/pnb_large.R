# ---- PNB Case Study: Full Bayesian Model Comparison ----
rm(list = ls())
library(dplyr)
library(tidyr)
library(rstan)
library(bridgesampling)
library(loo)
library(readr)

# --- 1. CONFIGURATION ---
ITER <- 2000
WARMUP <- 1000
CHAINS <- 4
CORES <- 4
SEED <- 1960
directory <- "d:/Projects/mergerBayes"
datadir <- file.path(directory, "data")
resultsdir <- file.path(directory, "results")
modelpath <- file.path(directory, "code", "bayesian.stan")
datafile <- file.path(datadir, "supreme_data.RData")

if (!dir.exists(resultsdir)) dir.create(resultsdir, recursive = TRUE)

# --- 2. LOAD DATA ---
load(datafile)
simdata <- simdata %>%
  filter(event_id == 1, year == 1960) %>%
  mutate(
    event_mkt = factor(year), tophold = factor(tophold),
    year = factor(year), marginInv = 1 / margin
  ) %>%
  filter(is.finite(marginInv) & margin >= 0.1 & shareIn >= 0.01) %>%
  droplevels()

# Calculate the 'Missing' outside share from the raw data (Large Firms Only)
# This effectively treats the removed fringe share as part of the outside option
min_s0_data <- simdata %>%
  group_by(year) %>%
  summarise(min_s0 = max(0.01, 1 - sum(shareIn)), .groups = "drop")

# RESCALE SHARES forces sum to 1.0 (Conditional Shares)
simdata <- simdata %>%
  group_by(year) %>%
  mutate(shareIn = shareIn / sum(shareIn)) %>%
  ungroup()

# --- 3. STRUCTURE DETECTION ---
n_mkts <- nlevels(simdata$event_mkt)
n_yrs <- nlevels(simdata$year)
is_single <- (n_mkts == 1 && n_yrs == 1)
avg_mInv <- mean(simdata$marginInv, na.rm = TRUE)

# --- 4. DATA LIST (SYNCHRONIZED WITH STAN) ---
# Renamed to sdata for brevity and to fix table reference issues
sdata <- list(
  N = nrow(simdata),
  shareIn = simdata$shareIn,
  marginInv = simdata$marginInv,
  rateDiff = simdata$rate_deposits - mean(simdata$rate_deposits),
  rateDiff_sd = 1.0,
  N_event = n_mkts,
  event = as.integer(simdata$event_mkt),
  N_tophold = nlevels(simdata$tophold),
  tophold = as.integer(simdata$tophold),
  N_year = n_yrs,
  year = as.integer(simdata$year),
  supply_model = 1,
  use_cutoff = if (is_single) 0L else 1L,
  grainsize = 1,
  N_market_year = n_yrs,
  market_year_idx = as.integer(simdata$year),
  K_s0 = if (is_single) 0L else 3L,
  X_s0 = if (is_single) matrix(0, n_yrs, 0) else matrix(0, n_yrs, 3),
  is_single_market = as.integer(n_mkts == 1),
  use_rho = 0L,
  use_hmt = 0L,
  fix_supply_intercept = 1L,
  avg_price_hmt = mean(simdata$rate_deposits),
  avg_margin_hmt = mean(simdata$margin),
  ssnip_hmt = 0.05,
  prior_sigma_share = 0.15,
  prior_sigma_margin = 2.0,
  prior_sigma_meanval_strat = 1.0, # RESTORED
  prior_sigma_meanval_fringe = 1.0,
  prior_alpha_mean = avg_mInv,
  prior_alpha_sd = avg_mInv * 5.0, # High flexibility
  s0_prior_mean = if (sum(min_s0_data$min_s0) == 0) qlogis(1 / nrow(simdata)) else qlogis(0.15),
  s0_prior_sd = 2.0,
  prior_sigma_alpha = if (is_single) 0.001 else 1.0,
  prior_sigma_beta_s0 = if (is_single) 0.001 else 1.0,
  prior_lkj = 2.0,
  implied_s0 = as.array(min_s0_data$min_s0),
  mky_to_event = as.array(rep(1L, n_yrs)),
  mky_to_year = as.array(seq_len(n_yrs))
)

# FIXED: Wrapping s0_offset in as.array() to fix the dimension mismatch
init_fun <- function() {
  # Smart Initialization for Large Firm Model: Start at the data-implied s0
  s0_start <- qlogis(pmax(0.01, min_s0_data$min_s0))
  
  list(
    mu_log_a = log(avg_mInv),
    s0_logit = as.array(s0_start),
    b_tophold_raw = rep(0, sdata$N_tophold),
    sigma_share = 0.05,
    sigma_margin = 0.5
  )
}

# --- 5. EXECUTION ---
cat("Compiling Stan Model...\n")
stan_mod <- stan_model(modelpath)

fit <- sampling(stan_mod,
  data = sdata, chains = CHAINS, cores = CORES,
  iter = ITER + WARMUP, warmup = WARMUP, seed = SEED,
  init = init_fun,
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)

# --- 6. RESULTS & SUMMARY TABLE ---
cat("Divergences: ", sum(rstan::get_divergent_iterations(fit)), "\n")
saveRDS(fit, file.path(resultsdir, "pnb_fit_1960.rds"))


# Define the Parameter Mapping
# Note: We use grepl or exact matches to handle how Stan labels
# vector parameters like s0[1]
prior_lookup <- data.frame(
  Parameter = c(
    "mu_log_a", "s0[1]", "mu_b_strat", "sigma_b_strat",
    "mu_b_fringe", "sigma_b_fringe", "sigma_share", "sigma_margin"
  ),
  Prior = c(
    # Alpha Prior (Log-Space center)
    sprintf(
      "Normal(log(%.2f), %.2f)",
      sdata$prior_alpha_mean, sdata$prior_alpha_sd
    ),

    # Outside Share Prior (Logit-Space)
    sprintf("Normal(%.2f, %.2f)", sdata$s0_prior_mean, sdata$s0_prior_sd),

    # Quality Means (Strategic vs Fringe)
    sprintf("Normal(0, %.2f)", sdata$prior_sigma_meanval_strat),
    sprintf("Normal(0, %.2f)", sdata$prior_sigma_meanval_fringe),

    # Quality Variations
    "Normal(0, 1.0)", # sigma_b_strat prior (Standardized)
    "Normal(0, 1.0)", # sigma_b_fringe prior (Standardized)

    # The "Slack" parameters (Critical for Calibration vs Estimation)
    sprintf("Normal(0, %.2f)", sdata$prior_sigma_share),
    sprintf("Normal(0, %.2f)", sdata$prior_sigma_margin)
  )
)


# Extract Posterior Estimates
post_summary <- as.data.frame(summary(fit,
  pars = c(
    "mu_log_a", "s0", "mu_b_strat", "sigma_b_strat",
    "mu_b_fringe", "sigma_b_fringe", "sigma_share", "sigma_margin"
  )
)$summary)

# Merge with Posterior Results
results_table <- post_summary %>%
  mutate(Parameter = rownames(.)) %>%
  left_join(prior_lookup, by = "Parameter") %>%
  select(Parameter, Prior,
    `Post Mean` = mean,
    `Post SD`   = sd,
    Rhat
  )

# 3. Display Results
cat("\n=== PNB 1960: STRUCTURAL PRIOR vs. POSTERIOR ===\n")
print(results_table, row.names = FALSE, digits = 3)


write.csv(
  results_table,
  file.path(resultsdir, "pnb_final_parameter_comparison.csv"),
  row.names = FALSE
)

# --- 7. PREDICTED VS ACTUAL SHARES ---
cat("\n=== PNB 1960: PREDICTED vs. ACTUAL SHARES ===\n")
post <- rstan::extract(fit)
# We use colMeans of pred_shareIn (assuming it exists in generated quantities)
# If not, we can calculate it from mu1
if ("pred_shareIn" %in% names(post)) {
  y_pred <- colMeans(post$pred_shareIn)
  share_comp <- simdata %>%
    mutate(
      Pred_Share = y_pred,
      Error = shareIn - Pred_Share
    ) %>%
    select(tophold, market_share = shareIn, Pred_Share, Error) %>%
    arrange(desc(market_share))

  print(head(share_comp, 10), row.names = FALSE, digits = 3)
  write_csv(share_comp, file.path(resultsdir, "pnb_share_fit.csv"))
} else {
  cat("pred_shareIn not found in posterior. Run with full generated quantities.\n")
}
