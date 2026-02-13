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
datafile <- file.path(datadir, "supreme_data.RData")

# Ensure output dir exists
if (!dir.exists(resultsdir)) dir.create(resultsdir, recursive = TRUE)

# --- LOAD DATA ---
# --- LOAD DATA ---
cat("Loading PNB Data from:", datafile, "\n")
load(datafile) # Loads 'simdata'

# PNB PANEL: All Years (1960-1965) as Separate Markets
simdata <- simdata %>%
  filter(event_id == 1) %>% # PNB Case Only
  droplevels() %>%
  mutate(
    event_mkt = year,
    tophold = factor(tophold),
    year = factor(year),
    marginInv = 1 / margin,
    loan_rate = 0
  ) %>%
  filter(year == 1960) %>% # SINGLE YEAR ONLY
  droplevels() %>%
  filter(is.finite(marginInv) & margin > 0) %>%
  filter(margin >= 0.1) # Surgical Outlier Removal

cat("Filtered PNB Panel Data (All Years, Margin >= 0.1):", nrow(simdata), "observations\n")
cat("Years (Markets):", paste(levels(simdata$year), collapse = ", "), "\n")
n_yrs <- nlevels(simdata$year)
cat("N Markets:", n_yrs, "\n")
is_single <- (n_yrs == 1)

# --- FRINGE SHARE ADJUSTMENT ---
# Remove firms < 1% share and add their total share to the lower bound of Outside Option
FRANGE_THRESH <- 0.0 # Keep all for Cutoff Analysis

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

# Calculate the 'Missing' outside share from the raw data (0.5% floor for stability)
min_s0_data <- simdata %>%
  group_by(year) %>%
  summarise(min_s0 = pmax(0.005, 1 - sum(shareIn)), .groups = "drop")

# RESCALE SHARES forces sum to 1.0 (Conditional Shares)
simdata <- simdata %>%
  group_by(event_mkt, year) %>%
  mutate(shareIn = shareIn / sum(shareIn)) %>%
  ungroup()

cat(sprintf(
  "After filtering (threshold: %.1f%%, removed: %d firms): %d observations\n",
  FRANGE_THRESH * 100,
  sum(fringe_shares$fringe_share > 0),
  nrow(simdata)
))
cat("Data-Driven Outside Floor (min_s0):", min_s0_data$min_s0, "\n")

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

# --- ALPHA PRIOR: MODEL-SPECIFIC (LEVEL-SPACE) ---
# Calculated inside the loop based on the Margins/Shares of the 1960 data.
# Centered roughly around 1/Mean(Margin) ~ 0.7 - 1.2.

# Stan Data Preparation (Template)
sdata_template <- list(
  use_cutoff = 0L, # DISABLED: Incompatible with absolute error likelihood
  N = nrow(simdata),
  shareIn = simdata$shareIn,
  marginInv = simdata$marginInv,
  rateDiff = (simdata$rate_deposits - mean(simdata$rate_deposits)) / sd(simdata$rate_deposits),
  rateDiff_sd = sd(simdata$rate_deposits),
  # log_assets REMOVED
  N_event = 1,
  event = rep(1L, nrow(simdata)),
  N_tophold = nlevels(simdata$tophold),
  tophold = as.integer(simdata$tophold),

  # Covariates - ASSETS REMOVED (Replaced by Random Effects)
  log_deposits = as.array(0.0),
  # REMOVED log_assets

  # rateDiff_sd already set above
  N_year = nlevels(simdata$year),
  year = as.integer(simdata$year),

  # NEW: Values defined above for consistency
  s0_prior_mean = qlogis(0.15),
  s0_prior_sd = 1.0,

  # HIERARCHICAL S0 DATA
  # Mapping: market_year -> (event, year)
  N_market_year = nlevels(simdata$year),
  market_year_idx = as.integer(simdata$year),
  mky_to_event = as.array(rep(1L, nlevels(simdata$year))), # All same event (PNB)
  mky_to_year = as.array(seq_len(nlevels(simdata$year))),  # Year index

  K_s0 = 0L, # No covariates for single-market case
  X_s0 = matrix(0, nlevels(simdata$year), 0),
  grainsize = 1,

  # --- SINGLE MARKET SINGLE PERIOD ---
  is_single_market = 1L, # 1960 Only
  use_rho = 0L,

  # Structural Constraints
  fix_supply_intercept = 1L,
  use_hmt = 0L,
  avg_price_hmt = mean(simdata$rate_deposits),
  avg_margin_hmt = mean(simdata$margin),
  ssnip_hmt = 0.05,

  # --- PRIOR SCALES ---
  # Tightened for Absolute Share Errors (matching bayesian.stan)
  prior_sigma_share = 0.01,
  prior_sigma_margin = 1.0,
  prior_sigma_meanval_strat = 1.0,
  prior_sigma_meanval_fringe = 1.0,
  # Structural Alpha
  prior_alpha_mean = 1.0,
  prior_alpha_sd = 1.0,

  # Hierarchical Priors
  prior_sigma_alpha = 1.0,
  prior_sigma_beta_s0 = 1.0,
  prior_lkj = 2.0,
  min_s0 = as.array(min_s0_data$min_s0)
)

# Compile Model
cat("Compiling Stan Model...\n")
stan_mod <- stan_model(modelpath)

models <- c("Bertrand", "Auction", "Cournot", "MonCom")

# --- HELPER: CALCULATE MODEL-SPECIFIC PRIORS ---
get_alpha_prior <- function(model_name, data) {
  # Data extremes (ROBUST: 5th/95th Percentile per user request)
  # Avoids outliers like margin=0.03 driving the prior
  s_low <- quantile(data$shareIn, 0.05)
  s_high <- quantile(data$shareIn, 0.95)
  m_low <- quantile(data$margin, 0.05)
  m_high <- quantile(data$margin, 0.95)
  s0_proxy <- 0.01

  if (model_name == "Bertrand") {
    # Alpha ~ 1 / (m * (1-s))
    a_low <- 1 / (m_high * (1 - s_low))
    a_high <- 1 / (m_low * (1 - s_high)) # m_low is robust now (e.g., ~0.5 not 0.03)
  } else if (model_name == "Auction") {
    # Alpha ~ -log(1-s) / (m * s)
    a_low <- -log(1 - s_low) / (m_high * s_low)
    a_high <- -log(1 - s_high) / (m_low * s_high)
  } else if (model_name == "Cournot") {
    # Alpha ~ (s0 + s) / (m * s0)
    a_low <- (s0_proxy + s_low) / (m_high * s0_proxy)
    a_high <- (s0_proxy + s_high) / (m_low * s0_proxy)
  } else { # MonCom
    # Alpha ~ 1 / m
    a_low <- 1 / m_high
    a_high <- 1 / m_low
  }

  # Construct Prior: N(mean, sd) to cover [a_low, a_high] with 95% mass
  p_mean <- (a_high + a_low) / 2
  p_sd <- (a_high - a_low) / 4

  return(list(mean = p_mean, sd = p_sd))
}

# --- BATCH EXECUTION FUNCTION ---
run_batch <- function(sdata, suffix, adapt_delta = 0.99) {
  hmt_status <- if (sdata$use_hmt == 1) "HMT-Constrained" else "Unconstrained"
  intercept_status <- if (sdata$fix_supply_intercept == 1) "Fixed-Intercept" else "Free-Intercept"
  
  msg <- sprintf(
    "\n\n>>> STARTING BATCH: %s [%s, %s] <<<\n",
    suffix, hmt_status, intercept_status
  )
  cat(msg)
  results <- list()

  for (m in 1:length(models)) {
    model_name <- models[m]
    
    cat("\n")
    cat("========================================\n")
    cat(sprintf("   MODEL %d of %d: %s\n", m, length(models), model_name))
    cat(sprintf("   Supply Model Code: %d\n", m))
    cat("========================================\n")
    cat("\n")

    sdata$supply_model <- as.integer(m)
    sdata$use_cutoff <- 0L # DISABLED for all models

    # Calculate Model-Specific Prior (Calibration Mode)
    # This aligns the prior with the FOCs of each model (Bertrand vs Auction)
    prior <- get_alpha_prior(model_name, simdata)
    sdata$prior_alpha_mean <- prior$mean
    sdata$prior_alpha_sd <- prior$sd
    cat(sprintf("   Alpha Prior: Normal(%.2f, %.2f)\n", prior$mean, prior$sd))


    # Safe initialization for s0_offset (matching run_pnb_single.R)
    init_fun <- function() {
      list(
        s0_offset = as.array(rep(qlogis(0.05) - qlogis(min_s0_data$min_s0[1]), sdata$N_market_year)),
        mu_log_a = log(sdata$prior_alpha_mean)
      )
    }

    fit <- sampling(
      stan_mod,
      data = sdata, chains = CHAINS, cores = CORES,
      iter = ITER + WARMUP, warmup = WARMUP, seed = SEED,
      init = init_fun,
      control = list(adapt_delta = adapt_delta, max_treedepth = 15)
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

    # Decoupled Params
    p_sigmam_mean <- mean(post$sigma_margin)
    p_sigmam_sd <- sd(post$sigma_margin)

    p_sigmas_raw <- if ("sigma_share" %in% names(post)) post$sigma_share else post$sigma_share_abs
    p_sigmas_mean <- mean(p_sigmas_raw)
    p_sigmas_sd <- sd(p_sigmas_raw)

    # Additional Parameters (Full Model)
    p_rho_mean <- if ("rho" %in% names(post)) mean(post$rho) else NA
    p_rho_sd <- if ("rho" %in% names(post)) sd(post$rho) else NA

    p_cutoff_mean <- if ("cutoff_share" %in% names(post)) mean(post$cutoff_share) else NA
    p_cutoff_sd <- if ("cutoff_share" %in% names(post)) sd(post$cutoff_share) else NA

    p_sigma_b_strat_mean <- if ("sigma_b_strat" %in% names(post)) mean(post$sigma_b_strat) else NA
    p_sigma_b_strat_sd <- if ("sigma_b_strat" %in% names(post)) sd(post$sigma_b_strat) else NA

    p_sigma_b_fringe_mean <- if ("sigma_b_fringe" %in% names(post)) mean(post$sigma_b_fringe) else NA
    p_sigma_b_fringe_sd <- if ("sigma_b_fringe" %in% names(post)) sd(post$sigma_b_fringe) else NA


    # Validation
    y1_pred <- colMeans(post$pred_shareIn)
    y2_pred <- colMeans(post$pred_marginInv)
    # Log-RMSE for shares (Structural fit check)
    rmse_share <- sqrt(mean((log(pmax(1e-9, sdata$shareIn)) - log(pmax(1e-9, y1_pred)))^2))
    rmse_margin <- sqrt(mean((sdata$marginInv - y2_pred)^2))

    results[[model_name]] <- list(
      Model = model_name, Divergences = divs, LogML = logml, LOOIC = looic,
      Alpha = p_alpha_mean, S0 = p_s0_mean,
      RMSE_Share = rmse_share, RMSE_Margin = rmse_margin,
      details = data.frame(
        Parameter = c(
          "Alpha", "S0", "Sigma_Share_Abs", "Sigma_Margin",
          "Rho", "Cutoff_Share", "Sigma_B_Strat",
          "Sigma_B_Fringe"
        ),
        Mean = c(
          p_alpha_mean, p_s0_mean, p_sigmas_mean, p_sigmam_mean,
          p_rho_mean, p_cutoff_mean,
          p_sigma_b_strat_mean, p_sigma_b_fringe_mean
        ),
        SD = c(
          p_alpha_sd, p_s0_sd, p_sigmas_sd, p_sigmam_sd, p_rho_sd,
          p_cutoff_sd, p_sigma_b_strat_sd,
          p_sigma_b_fringe_sd
        )
      )
    )
  }

  # --- SUMMARY TABLES ---
  df_res <- bind_rows(lapply(results, function(x) x[names(x) != "details"]))
  max_logml <- max(df_res$LogML, na.rm = TRUE)
  if (is.infinite(max_logml)) max_logml <- -1e10 # Fallback

  df_res <- df_res %>%
    mutate(
      diff_logml = ifelse(is.na(LogML), NA, LogML - max_logml),
      bayes_factor_vs_best = ifelse(is.na(diff_logml), NA, exp(diff_logml)),
      posterior_prob = ifelse(is.na(diff_logml), 0, exp(diff_logml) / sum(exp(diff_logml), na.rm = TRUE))
    ) %>%
    arrange(desc(LogML))

  outfile_res <- file.path(resultsdir, paste0("pnb_model_results", suffix, ".csv"))
  write_csv(df_res, outfile_res)

  # Parameter Table
  priors <- data.frame(
    Parameter = c(
      "Alpha", "S0", "Sigma_Share_Abs", "Sigma_Margin",
      "Rho", "Cutoff_Share", "Sigma_B_Strat",
      "Sigma_B_Fringe"
    ),
    Prior_Desc = c(
      sprintf("Normal(%.2f, %.2f)", sdata$prior_alpha_mean, sdata$prior_alpha_sd),
      sprintf("Logit(%.2f, %.2f)", sdata$s0_prior_mean, sdata$s0_prior_sd),
      sprintf("Normal(0, %.3f)", sdata$prior_sigma_share),
      sprintf("Normal(0, %.2f)", sdata$prior_sigma_margin),
      "LKJ(2.0)", "Beta(3,100)",
      "Normal(0, 5)", "Normal(0, 2)"
    ),
    Prior_Mean = c(
      sprintf("%.2f", sdata$prior_alpha_mean),
      sprintf("%.2f", plogis(sdata$s0_prior_mean)),
      "0.000", "0.000", "0.000", "0.03", "0.0", "0.0"
    ),
    Prior_SD = c(
      sprintf("%.2f", sdata$prior_alpha_sd),
      sprintf("%.2f", sdata$s0_prior_sd),
      sprintf("%.3f", sdata$prior_sigma_share),
      sprintf("%.2f", sdata$prior_sigma_margin),
      "0.50", "0.02", "5.0", "0.2"
    )
  )
  priors$Prior_Col <- paste0(priors$Prior_Mean, " (", priors$Prior_SD, ")")
  param_table <- priors %>% select(Parameter, Prior = Prior_Col)

  # Dynamically process only the models that were actually run
  run_models <- intersect(models, names(results))

  for (m in run_models) {
    det <- results[[m]]$details
    if (!is.null(det)) {
      # Rename 'Formatted' column to the model name
      det_col <- det %>%
        mutate(!!m := sprintf("%.3f (%.3f)", Mean, SD)) %>%
        select(Parameter, !!m)

      param_table <- left_join(param_table, det_col, by = "Parameter")
    }
  }

  outfile_param <- file.path(resultsdir, paste0("pnb_parameter_comparison", suffix, ".csv"))
  write_csv(param_table, outfile_param)
}

# --- EXECUTE BATCHES ---

# 3. Fixed Supply Intercept Run (Strict Structural)
sdata_fixed <- sdata_template
sdata_fixed$fix_supply_intercept <- 1L
# min_s0 is already set in sdata_template from data
run_batch(sdata_fixed, "_fixed")
