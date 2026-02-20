# ---- Load Libraries ----
rm(list = ls())

library(dplyr)
library(tidyr)
library(antitrust)
library(ggplot2)
library(kableExtra)
library(stringr)
library(readr)
library(rstan)
library(bridgesampling)
library(jsonlite)
library(loo)

# Helper for logging
log_msg <- function(...) {
  cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), ..., "\n")
  flush.console()
}

# Process command line arguments
args <- commandArgs(trailingOnly = TRUE)
thismodel <- as.numeric(args[1])
if (is.na(thismodel)) thismodel <- 1

chain_count <- as.numeric(args[2])
if (is.na(chain_count)) chain_count <- 4

thread_count <- as.numeric(args[3])
if (is.na(thread_count)) thread_count <- 1

# Toggle Cutoff (1=yes)
use_cutoff <- as.numeric(args[4])
if (is.na(use_cutoff)) use_cutoff <- 0

# Toggle HMT Constraint (1=yes, default 0)
use_hmt_arg <- as.numeric(args[5])
if (is.na(use_hmt_arg)) use_hmt_arg <- 0

# Data Subsampling Fraction (0 to 1, default 1.0)
data_frac <- as.numeric(args[6])
if (is.na(data_frac)) data_frac <- 1.0

# Year Filter (Single "2014" or Range "2014-2016")
filter_year_str <- as.character(args[7])
if (is.na(filter_year_str) || filter_year_str == "NA") filter_year_str <- "2014-2015"

if (grepl("-", filter_year_str)) {
  parts <- as.numeric(unlist(strsplit(filter_year_str, "-")))
  start_year <- as.numeric(as.character(parts[1]))
  end_year <- as.numeric(as.character(parts[2]))
} else {
  start_year <- as.numeric(filter_year_str)
  end_year <- start_year
}


# Set up Stan parallelization
options(mc.cores = chain_count)
rstan_options(auto_write = TRUE)
Sys.setenv(STAN_NUM_THREADS = thread_count)
rstan_options(threads_per_chain = thread_count)

model_name <- c("bertrand", "2nd", "cournot", "moncom")
log_msg(paste("Running Model:", model_name[thismodel], "| Chains:", chain_count, "| Threads:", thread_count))

directory <- file.path(Sys.getenv("HOME"), "Projects", "mergerBayes")
datadir <- file.path(directory, "data")
resultsdir <- file.path(directory, "results")
modelpath <- file.path(directory, "code", "bayesian.stan")

# Default outfile (Panel)
outfile <- file.path(resultsdir, paste0("stan_hhiperform_", model_name[thismodel], ".RData"))
outfile_summary <- file.path(resultsdir, paste0("summary_", model_name[thismodel], ".rds"))

# Ensure results directory exists
if (!dir.exists(resultsdir)) dir.create(resultsdir, recursive = TRUE)

# Load Data
log_msg("Loading data...")
load(file = file.path(datadir, "banksimdata.RData"))

# Filter for 2014-2015 (Standard Panel Filter)
# NOTE: If running PNB, you might want to adjust this filter manually or add logic.
# For now, we assume simdata contains the relevant observations.
# Filter using command line arguments
simdata <- simdata %>% filter(year >= start_year & year <= end_year)
log_msg(paste("Data filtered. Rows:", nrow(simdata)))

# Clean and process data
simdata <- simdata %>%
  filter(deposit_share > 0) %>%
  mutate(
    event_mkt = interaction(event_id, fedmkt, drop = TRUE),
    tophold = factor(tophold),
    year = factor(year),
    shareIn = deposit_share,
    margin = selected_margin,
    rate_deposits = rate_call_report,
    rate_loans = 0 # Proxy
  ) %>%
  mutate(margin = ifelse(margin <= 0, NA, margin)) %>%
  filter(!is.na(margin)) %>%
  droplevels() %>%
  arrange(event_mkt, year, tophold)

# ------------------------------------------------------------------------------
# FILTER SMALL MARKETS (Mergers to Monopoly / Monopolies)
# ------------------------------------------------------------------------------
# Count firms per market-year
mkt_counts <- simdata %>%
  group_by(event_mkt, year) %>%
  summarise(n_firms = n(), .groups = "drop")

# Identify markets where ANY year has <= 2 firms (Monopoly or Duopoly)
# Note: A merger to monopoly usually implies starting with 2 firms.
bad_markets <- mkt_counts %>%
  filter(n_firms <= 2) %>%
  pull(event_mkt) %>%
  unique()

if (length(bad_markets) > 0) {
  log_msg(paste("Dropping", length(bad_markets), "markets with <= 2 firms (Monopoly/Duopoly risk)."))
  simdata <- simdata %>% filter(!event_mkt %in% bad_markets)
}

# Re-level factors after filtering
simdata <- simdata %>%
  mutate(
    event_mkt = droplevels(event_mkt),
    tophold = droplevels(tophold),
    year = as.numeric(as.character(droplevels(year)))
  )

# ------------------------------------------------------------------------------
# DATA REDUCTION (Optional)
# ------------------------------------------------------------------------------

# 1. Year Filter
if (start_year > 0) {
  if (start_year == end_year) {
    log_msg(paste("Filtering for Year:", start_year))
    simdata <- simdata %>% filter(year == start_year)
  } else {
    log_msg(paste("Filtering for Year Range:", start_year, "-", end_year))
    simdata <- simdata %>% filter(year %in% start_year:end_year)
  }
}


# 2. Random Subsample (Market-Level)
if (data_frac < 1.0 && data_frac > 0) {
  log_msg(paste("Subsampling", data_frac * 100, "% of markets..."))
  all_markets <- levels(droplevels(simdata$event_mkt))
  n_keep <- max(5, round(length(all_markets) * data_frac))

  set.seed(42) # Ensure reproducible subsamples
  keep_markets <- sample(all_markets, n_keep)

  simdata <- simdata %>% filter(event_mkt %in% keep_markets)
  log_msg(paste("Reduced to", n_keep, "markets."))
}

# Re-level factors after filtering/subsampling
simdata <- simdata %>%
  mutate(
    event_mkt = droplevels(event_mkt),
    tophold = droplevels(tophold),
    year = factor(year)
  )


# ------------------------------------------------------------------------------
# PNB / SINGLE MARKET DETECTION
# ------------------------------------------------------------------------------
n_markets <- nlevels(simdata$event_mkt)
is_single_market <- ifelse(n_markets == 1, 1, 0)

if (is_single_market == 1) {
  log_msg(">>> DETECTED SINGLE MARKET MODE (PNB) <<<")
  outfile <- file.path(resultsdir, paste0("stan_PNB_", model_name[thismodel], ".RData"))
  outfile_summary <- file.path(resultsdir, paste0("summary_PNB_", model_name[thismodel], ".rds"))
} else {
  log_msg(paste("Panel Mode Detected. N_markets:", n_markets))
}

if (nrow(simdata) < 10) stop("Too few observations!")

# ------------------------------------------------------------------------------
# FRINGE HANDLING (Consistent with PNB/Stan Logic)
# ------------------------------------------------------------------------------
# Define Fringe: Firms with < 1.0% share
FRANGE_THRESH <- 0.01

# 1. Calculate Fringe Share per Market (to pass as min_s0)
# Step 1: Sum fringe share per market-year
# Step 2: Average across years to get a single stable prior anchor per market
fringe_summary <- simdata %>%
  group_by(event_mkt, year) %>%
  summarise(
    yr_fringe = sum(shareIn[shareIn < FRANGE_THRESH], na.rm = TRUE),
    .groups = "drop_last"
  ) %>%
  summarise(
    fringe_share = mean(yr_fringe, na.rm = TRUE), # Average over years
    .groups = "drop"
  ) %>%
  complete(event_mkt, fill = list(fringe_share = 0)) %>%
  arrange(event_mkt)

min_s0_vec <- fringe_summary$fringe_share

# 2. Filter & Rescale Data
# 2. Filter & Rescale Data
log_msg(paste("Removing firms with share <", FRANGE_THRESH * 100, "%"))
simdata <- simdata %>%
  filter(shareIn >= FRANGE_THRESH) %>%
  # group_by(event_mkt, year) %>% # RESCALING DISABLED (Hard Floor Logic)
  # mutate(shareIn = shareIn / sum(shareIn)) %>%
  # ungroup() %>%
  droplevels() # Important: Re-level to ensure indexing matches

# Re-align min_s0 vector to new filtered levels if any markets dropped (unlikely but safe)
# The levels(simdata$event_mkt) might have changed if a market was purely fringe (invalid)
if (nlevels(simdata$event_mkt) != length(min_s0_vec)) {
  # Re-match
  current_levels <- levels(simdata$event_mkt)
  min_s0_vec <- fringe_summary %>%
    filter(event_mkt %in% current_levels) %>%
    arrange(factor(event_mkt, levels = current_levels)) %>%
    pull(fringe_share)
}

# ------------------------------------------------------------------------------
# PREPARE HMT INPUTS
# ------------------------------------------------------------------------------
avg_price_hmt <- mean(simdata$rate_deposits, na.rm = TRUE)
avg_margin_hmt <- mean(simdata$margin, na.rm = TRUE)
ssnip_hmt <- 0.05

# ------------------------------------------------------------------------------
# PREPARE DATA LIST
# ------------------------------------------------------------------------------

# Hierarchical Summaries
eventdata <- simdata %>%
  group_by(event_mkt) %>%
  summarise(deposit_total_market = mean(deposit_total_market, na.rm = TRUE))
topholddata <- simdata %>%
  group_by(tophold) %>%
  summarise(total_deposits = mean(deposits, na.rm = TRUE))

simdata <- simdata %>% mutate(market_year = interaction(event_mkt, year, drop = TRUE))
simdata <- simdata %>% mutate(market_year_idx = as.numeric(market_year))
N_market_year <- length(unique(simdata$market_year))

# Build market-year -> event and market-year -> year mappings
mky_to_event <- integer(N_market_year)
mky_to_year  <- integer(N_market_year)
event_int <- as.integer(simdata$event_mkt)
year_int  <- as.integer(simdata$year)
for (i in seq_len(nrow(simdata))) {
  mky_to_event[simdata$market_year_idx[i]] <- event_int[i]
  mky_to_year[simdata$market_year_idx[i]]  <- year_int[i]
}

# Implied outside share per market-year (from fringe firms removed earlier)
implied_s0 <- numeric(N_market_year)
for (m in seq_len(N_market_year)) {
  idx <- which(simdata$market_year_idx == m)
  implied_s0[m] <- max(0.01, min(0.99, 1.0 - sum(simdata$shareIn[idx])))
}

stan_data <- list(
  # --- Standard Inputs ---
  N = nrow(simdata),
  shareIn = simdata$shareIn,
  marginInv = 1 / simdata$margin,
  rateDiff = as.numeric(scale(simdata$rate_deposits, center = TRUE, scale = TRUE)),
  rateDiff_sd = sd(simdata$rate_deposits, na.rm = TRUE),

  # --- Indexing ---
  N_event = nlevels(simdata$event_mkt),
  event = as.integer(simdata$event_mkt),
  N_tophold = nlevels(simdata$tophold),
  tophold = as.integer(simdata$tophold),
  N_year = nlevels(simdata$year),
  year = as.integer(simdata$year),
  supply_model = as.integer(thismodel),
  use_cutoff = as.integer(use_cutoff),
  grainsize = max(100, round(nrow(simdata) / (4 * chain_count))),
  N_market_year = N_market_year,
  market_year_idx = as.integer(simdata$market_year_idx),

  # --- S0 Covariates: Intercept + log(Deposits) + log(Assets) ---
  K_s0 = 3L,
  X_s0 = simdata %>%
    group_by(market_year) %>%
    summarise(
      Intercept = 1,
      log_dep = mean(log(total_deposits), na.rm = TRUE),
      log_ast = mean(log(total_assets), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    select(Intercept, log_dep, log_ast) %>%
    as.matrix(),

  # --- Market-year mappings ---
  mky_to_event = as.array(mky_to_event),
  mky_to_year  = as.array(mky_to_year),

  # --- Flags ---
  is_single_market = as.integer(is_single_market),
  use_rho = 1L,
  use_hmt = as.integer(use_hmt_arg),

  # --- HMT inputs ---
  avg_price_hmt = avg_price_hmt,
  avg_margin_hmt = avg_margin_hmt,
  ssnip_hmt = ssnip_hmt,

  # --- Prior Scales ---
  prior_sigma_share = 0.005,
  prior_sigma_margin = 1.0 * mean(1 / simdata$margin, na.rm = TRUE),
  prior_sigma_meanval_strat = 1.5,
  prior_sigma_meanval_fringe = 0.3,
  prior_alpha_mean = 4.0,
  prior_alpha_sd = 3.0,
  s0_prior_mean = -1.0,
  s0_prior_sd = 1.0,
  prior_sigma_alpha = 1.0,
  prior_sigma_beta_s0 = 1.0,
  prior_lkj = 4.0,
  prior_cutoff_alpha = 3.0,
  prior_cutoff_beta = 100.0,

  # --- Implied outside share ---
  implied_s0 = as.array(implied_s0)
)

# Export for Debugging
write_json(stan_data, file.path(datadir, "stan_data.json"), pretty = TRUE, auto_unbox = TRUE)

log_msg("Data preparation complete. Compiling model...")

# Compile
model <- stan_model(modelpath)

# Sampling
target_adapt_delta <- 0.99
# if (thismodel == 3) target_adapt_delta <- 0.99
# If PNB mode, use higher adapt_delta
# if (is_single_market == 1) target_adapt_delta <- 0.99

log_msg(paste("Starting sampling (4000 iter, thin=2) | adapt_delta:", target_adapt_delta))

fit <- sampling(
  model,
  data = stan_data,
  chains = chain_count,
  iter = 4000,
  warmup = 1000,
  thin = 2,
  control = list(adapt_delta = target_adapt_delta, max_treedepth = 13)
)
log_msg("Sampling complete.")

# ------------------------------------------------------------------------------
# SUMMARIES & SAVING
# ------------------------------------------------------------------------------
log_msg("Generating summaries...")

# Parameters to extract (matched to current bayesian.stan)
base_pars <- c("a_event", "s0", "sigma_share", "sigma_margin", "rho", "mu_log_a")
if (is_single_market == 0) {
  base_pars <- c(base_pars, "b_event")
}
if (stan_data$use_cutoff == 1) base_pars <- c(base_pars, "cutoff_share")
if (nlevels(simdata$year) > 1) base_pars <- c(base_pars, "year_effect_s0")

fit_sum <- summary(fit, pars = base_pars)$summary

# Save SUMMARY immediately
log_msg("Saving summary file...")
saveRDS(list(summary = fit_sum, stan_data = stan_data), file = outfile_summary)

# LOO
log_msg("Running LOO...")
loo_result <- tryCatch(
  {
    loo(fit, cores = thread_count)
  },
  error = function(e) {
    log_msg("LOO Failed:", e$message)
    return(NULL)
  }
)
if (!is.null(loo_result)) log_msg(paste("LOO ELPD:", loo_result$estimates["elpd_loo", "Estimate"]))

# Bridge
# Bridge
log_msg("Attempting Bridge Sampler (High Iteration)...")
bridge <- tryCatch(
  {
    fit@stanmodel <- model
    # First attempt: Standard Normal with more iterations
    bridge_sampler(fit, silent = FALSE, cores = thread_count, maxiter = 10000, method = "normal")
  },
  error = function(e) {
    log_msg("Bridge (Normal) Failed:", e$message)
    log_msg("Attempting Bridge Sampler (Warp3)...")
    tryCatch(
      {
        # Second attempt: Warp3 method (slower but robust)
        bridge_sampler(fit, silent = FALSE, cores = thread_count, maxiter = 5000, method = "warp3")
      },
      error = function(e2) {
        log_msg("Bridge (Warp3) Failed:", e2$message)
        return(NULL)
      }
    )
  }
)


# Final Save
log_msg(paste("Saving final results to:", outfile))
tryCatch(
  {
    save(fit, loo_result, bridge, fit_sum, file = outfile, compress = "xz")
    log_msg("Full results saved.")
  },
  error = function(e) {
    log_msg("ERROR saving full results:", e$message)
    log_msg("Attempting fallback save...")
    posteriors <- rstan::extract(fit)
    save(posteriors, loo_result, bridge, fit_sum, file = outfile, compress = "xz")
  }
)

log_msg("Script finished successfully.")
