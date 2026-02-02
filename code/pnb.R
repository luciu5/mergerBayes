# ---- PNB Case Study: Full Bayesian Model Comparison ----
# Single-year analysis using the actual bank-level data from the Supreme Court case
rm(list = ls())

library(dplyr)
library(tidyr)
library(rstan)
library(bridgesampling)
library(loo)
library(readr)

# --- CONFIGURATION ---
ITER <- 2000
WARMUP <- 1000
CHAINS <- 4
CORES <- 4
SEED <- 1960 # Data year

# Paths
directory <- "d:/Projects/mergerBayes"
datadir <- file.path(directory, "data")
resultsdir <- file.path(directory, "results")
modelpath <- file.path(directory, "code", "bayesian.stan")
datafile <- file.path(datadir, "pnb_case_data.csv")

# Ensure output dir exists
if (!dir.exists(resultsdir)) dir.create(resultsdir, recursive = TRUE)

# --- LOAD DATA ---
cat("Loading PNB Data from:", datafile, "\n")
simdata <- read_csv(datafile, show_col_types = FALSE)

cat("Loaded", nrow(simdata), "observations\n")

# Pre-processing
simdata <- simdata %>%
  mutate(
    event_mkt = factor(1),
    tophold = factor(tophold),
    year = factor(year),
    marginInv = 1 / margin,
    loan_rate = 0
  ) %>%
  filter(is.finite(marginInv) & margin > 0)

cat("After initial filtering:", nrow(simdata), "observations\n")

# --- FRINGE SHARE ADJUSTMENT ---
# Remove firms < 1% share and add their total share to the lower bound of Outside Option
FRANGE_THRESH <- 0.01

# Calculate fringe share per market
fringe_shares <- simdata %>%
  group_by(event_mkt) %>%
  summarise(
    fringe_share = sum(shareIn[shareIn < FRANGE_THRESH]),
    .groups = "drop"
  )

# Filter out small firms
simdata <- simdata %>%
  filter(shareIn >= FRANGE_THRESH)
# filter(shareIn >= 0) # Keep all

# RESCALE SHARES forces sum to 1.0
# DISABLED: No Rescale + Hard Floor in Stan is the stable strategy.
# simdata <- simdata %>%
#   group_by(event_mkt, year) %>%
#   mutate(shareIn = shareIn / sum(shareIn)) %>%
#   ungroup()

cat("After removing fringe (<1%):", nrow(simdata), "observations\n")
cat("Fringe Share (min_s0):", fringe_shares$fringe_share, "\n")

# --- GENERATE DATA SUMMARY TABLE ---
cat("\n=== GENERATING DATA SUMMARY TABLE ===\n")
data_summary <- simdata %>%
  summarise(
    Variable = c("Margin", "shareIn", "rate_deposits"),
    Mean = c(mean(margin), mean(shareIn), mean(rate_deposits)),
    SD = c(sd(margin), sd(shareIn), sd(rate_deposits)),
    Min = c(min(margin), min(shareIn), min(rate_deposits)),
    Max = c(max(margin), max(shareIn), max(rate_deposits))
  )

print(data_summary)
write_csv(data_summary, file.path(resultsdir, "pnb_data_summary.csv"))

# Calculate Data Driven Priors
sd_logshare <- sd(log(simdata$shareIn))
sd_marginInv <- sd(simdata$marginInv)

# Aggregates for Covariates
bank_assets <- simdata$total_assets

# Stan Data Preparation (Template)
sdata_template <- list(
  use_cutoff = 0L, # DISABLED: Vanilla model (no Split Prior)
  N = nrow(simdata),
  shareIn = simdata$shareIn,
  marginInv = simdata$marginInv,
  rateDiff = as.numeric(scale(simdata$rate_deposits)),
  # log_assets REMOVED
  N_event = 1,
  event = rep(1L, nrow(simdata)),
  N_tophold = nlevels(simdata$tophold),
  tophold = as.integer(simdata$tophold),

  # Covariates - ASSETS REMOVED (Replaced by Random Effects)
  # Covariates - ASSETS REMOVED
  log_deposits = as.array(0.0),
  # REMOVED log_assets

  rateDiff_sd = sd(simdata$rate_deposits),
  N_year = 1,
  year = rep(1L, nrow(simdata)),
  N_market_year = 1,
  market_year_idx = rep(1L, nrow(simdata)),
  grainsize = 1,

  # --- SINGLE MARKET MODE ---
  is_single_market = 1L,
  use_hmt = 0L, # Default off
  fix_supply_intercept = 1L, # Enforce Supply Intercept = 0 (Calibration Style)
  min_s0 = as.array(fringe_shares$fringe_share), # PASS FRINGE SHARE LB
  avg_price_hmt = mean(simdata$rate_deposits),
  avg_margin_hmt = mean(simdata$margin),
  ssnip_hmt = 0.05,

  # --- PRIOR SCALES ---
  # TEST: Tight Share (Hard Data) vs Relaxed Margin (Soft Data)
  prior_sigma_share = 0.10,
  prior_sigma_margin = 1.0, # Multiplier for margin prior (Very loose)
  prior_sigma_meanval_strat = 1.5,
  prior_sigma_meanval_fringe = 0.2
)

# Compile Model
cat("Compiling Stan Model...\n")
stan_mod <- stan_model(modelpath)

models <- c("Bertrand", "Auction", "Cournot", "MonCom")

# --- BATCH EXECUTION FUNCTION ---
run_batch <- function(sdata, suffix) {
  cat(sprintf("\n\n>>> STARTING BATCH: %s <<<\n", ifelse(suffix == "", "Standard", "HMT Constrained")))
  results <- list()

  for (m in 1:4) {
    model_name <- models[m]
    cat(sprintf("\n=== RUNNING MODEL: %s (Suffix: %s) ===\n", model_name, suffix))

    sdata$supply_model <- as.integer(m)

    adapt_delta <- 0.99 # Increased back to 0.99 (0.95 caused 2000+ divs)
    # if (m == 3) adapt_delta <- 0.95

    fit <- sampling(
      stan_mod,
      data = sdata, chains = CHAINS, cores = CORES,
      iter = ITER + WARMUP, warmup = WARMUP, seed = SEED,
      control = list(adapt_delta = adapt_delta, max_treedepth = 12)
    )

    # Diagnostics
    sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
    divs <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
    cat(sprintf("Divergences: %d\n", divs))

    # Bridge Sampling
    logml_res <- tryCatch(
      {
        bridge_sampler(fit, silent = TRUE, maxiter = 5000)
      },
      error = function(e) {
        NULL
      }
    )
    logml <- if (!is.null(logml_res)) logml_res$logml else NA
    cat(sprintf("LogML: %.2f\n", logml))

    # LOO
    loo_res <- tryCatch(
      {
        loo(fit)
      },
      error = function(e) {
        NULL
      }
    )
    looic <- if (!is.null(loo_res)) loo_res$estimates["looic", "Estimate"] else NA
    cat(sprintf("LOOIC: %.2f\n", looic))

    # Extract
    post <- rstan::extract(fit)
    p_alpha_mean <- mean(post$a_event)
    p_alpha_sd <- sd(post$a_event)
    p_s0_mean <- mean(plogis(post$s0))
    p_s0_sd <- sd(plogis(post$s0)) # Probability Scale
    p_rho_mean <- mean(post$rho_gen)
    p_rho_sd <- sd(post$rho_gen)
    p_sigmam_mean <- mean(post$sigma_margin)
    p_sigmam_sd <- sd(post$sigma_margin)
    p_sigmas_mean <- mean(post$sigma_logshare)
    p_sigmas_sd <- sd(post$sigma_logshare)
    p_cutoff_mean <- mean(post$cutoff_share)
    p_cutoff_sd <- sd(post$cutoff_share)

    # Validation
    y1_pred <- colMeans(post$pred_logshareIn)
    y2_pred <- colMeans(post$pred_marginInv)
    rmse_share <- sqrt(mean((log(sdata$shareIn) - y1_pred)^2))
    rmse_margin <- sqrt(mean((sdata$marginInv - y2_pred)^2))

    results[[model_name]] <- list(
      Model = model_name, Divergences = divs, LogML = logml, LOOIC = looic,
      Alpha = p_alpha_mean, S0 = p_s0_mean, Rho = p_rho_mean,
      RMSE_Share = rmse_share, RMSE_Margin = rmse_margin,
      details = data.frame(
        Parameter = c("Alpha", "S0", "Sigma_Share", "Sigma_Margin", "Rho", "Cutoff"),
        Mean = c(p_alpha_mean, p_s0_mean, p_sigmas_mean, p_sigmam_mean, p_rho_mean, p_cutoff_mean),
        SD = c(p_alpha_sd, p_s0_sd, p_sigmas_sd, p_sigmam_sd, p_rho_sd, p_cutoff_sd)
      )
    )
  }

  # --- SUMMARY TABLES ---
  df_res <- bind_rows(lapply(results, function(x) x[names(x) != "details"]))
  max_logml <- max(df_res$LogML, na.rm = TRUE)
  df_res <- df_res %>%
    mutate(
      diff_logml = LogML - max_logml,
      bayes_factor_vs_best = exp(diff_logml),
      posterior_prob = exp(diff_logml) / sum(exp(diff_logml), na.rm = TRUE)
    ) %>%
    arrange(desc(LogML))

  outfile_res <- file.path(resultsdir, paste0("pnb_model_results", suffix, ".csv"))
  write_csv(df_res, outfile_res)

  # Parameter Table
  priors <- data.frame(
    Parameter = c("Alpha", "S0", "Sigma_Share", "Sigma_Margin", "Rho", "Cutoff"),
    Prior_Desc = c("LogNorm(0,0.5)", "Logit(-0.7,1)", sprintf("Normal(0, %.2f)", sd_logshare), sprintf("Normal(0, %.2f)", sd_marginInv), "LKJ(4)", "Beta(3,100)"),
    Prior_Mean = c("1.13", "0.33", "0.00", "0.00", "0.00", "0.03"),
    Prior_SD = c("0.60", "0.20", sprintf("%.2f", sd_logshare), sprintf("%.2f", sd_marginInv), "0.25", "0.02")
  )
  priors$Prior_Col <- paste0(priors$Prior_Mean, " (", priors$Prior_SD, ")")
  param_table <- priors %>% select(Parameter, Prior = Prior_Col)

  for (m in models) {
    det <- results[[m]]$details
    param_table <- left_join(param_table, det %>% mutate(Formatted = sprintf("%.3f (%.3f)", Mean, SD)) %>% select(Parameter, !!m := Formatted), by = "Parameter")
  }

  outfile_param <- file.path(resultsdir, paste0("pnb_parameter_comparison", suffix, ".csv"))
  write_csv(param_table, outfile_param)
}

# --- EXECUTE BATCHES ---

# 1. Tight Share / Relaxed Margin Batch
sdata_tight <- sdata_template
sdata_tight$prior_sigma_margin <- 1.0 * mean(simdata$marginInv, na.rm = TRUE)
run_batch(sdata_tight, "_tight")

# 2. Relaxed Prior (0.30) - DISABLED
# sdata_relaxed <- sdata_template
# sdata_relaxed$prior_sigma_margin <- 0.30 * mean(simdata$marginInv, na.rm = TRUE)
# run_batch(sdata_relaxed, "_relaxed")

# 3. Fixed Supply Intercept Run (Strict Prop 1)
# sdata_fixed <- sdata_template
# sdata_fixed$fix_supply_intercept <- 1L
# run_batch(sdata_fixed, "_fixed")
