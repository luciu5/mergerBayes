# ---- PNB Case Study: Legacy Model Run ----
# Running bayesian_pnb.stan to check for convergence
rm(list = ls())

library(dplyr)
library(tidyr)
library(rstan)
library(readr)

# --- CONFIGURATION ---
ITER <- 2000
WARMUP <- 1000
CHAINS <- 4
CORES <- 4
SEED <- 1960

# Paths
directory <- "d:/Projects/mergerBayes"
datadir <- file.path(directory, "data")
resultsdir <- file.path(directory, "results")
modelpath <- file.path(directory, "code", "bayesian_single.stan")
datafile <- file.path(datadir, "supreme_data.RData")

# Ensure output dir exists
if (!dir.exists(resultsdir)) dir.create(resultsdir, recursive = TRUE)

# --- LOAD DATA ---
cat("Loading PNB Data from:", datafile, "\n")
load(datafile) # Loads 'simdata'

# Filter for the Supreme Court Case (1960)
simdata <- simdata %>%
    filter(event_id == 1, year == 1960) %>%
    mutate(
        tophold = factor(tophold),
        year = factor(year),
        marginInv = 1 / margin,
        rateDiff = (rate_deposits - mean(rate_deposits)) / sd(rate_deposits)
    ) %>%
    filter(is.finite(marginInv) & margin >= 0.1) %>%
    droplevels()

# Calculate min_s0 (outside share floor: 0.5% as residual)
min_s0_val <- pmax(0.005, 1 - sum(simdata$shareIn))

# Rescale shares to sum to 1
simdata <- simdata %>%
    mutate(shareIn = shareIn / sum(shareIn))

cat("Observations:", nrow(simdata), "\n")
cat("min_s0:", min_s0_val, "\n")

# --- STAN DATA ---
sdata <- list(
    N = nrow(simdata),
    shareIn = simdata$shareIn,
    marginInv = simdata$marginInv,
    rateDiff = simdata$rateDiff,
    rateDiff_sd = sd(simdata$rate_deposits),
    N_event = 1,
    event = rep(1L, nrow(simdata)),
    N_tophold = nlevels(simdata$tophold),
    tophold = as.integer(simdata$tophold),
    log_deposits = as.array(0.0), # Scalar array for N_event=1

    supply_model = 1, # Bertrand
    use_cutoff = 0L,
    grainsize = 1,
    is_single_market = 1L,
    use_hmt = 0L,
    fix_supply_intercept = 1L,
    avg_price_hmt = mean(simdata$rate_deposits),
    avg_margin_hmt = mean(simdata$margin),
    ssnip_hmt = 0.05,
    prior_sigma_share = 0.01,
    prior_sigma_margin = 1.0,
    prior_sigma_meanval_strat = 1.0,
    prior_sigma_meanval_fringe = 1.0,
    prior_alpha_mean = 1.0,
    prior_alpha_sd = 1.0,
    min_s0 = as.array(min_s0_val)
)

# --- COMPILE & RUN ---
cat("Compiling Stan Model:", modelpath, "\n")
stan_mod <- stan_model(modelpath)

# Safe initialization for s0_offset to avoid starting at floor
init_fun <- function() {
    list(
        s0_offset = as.array(qlogis(0.05) - qlogis(min_s0_val)), # Start at 5% share
        mu_log_a = log(1.0)
    )
}

cat("Starting Sampler...\n")
fit <- sampling(
    stan_mod,
    data = sdata,
    init = init_fun,
    chains = CHAINS,
    cores = CORES,
    iter = ITER + WARMUP,
    warmup = WARMUP,
    seed = SEED,
    control = list(adapt_delta = 0.95, max_treedepth = 15)
)

# --- RESULTS ---
print(fit, pars = c("mu_log_a", "s0", "sigma_share_abs", "sigma_margin"))

# Save summary
sum_fit <- summary(fit)$summary
write.csv(sum_fit, file.path(resultsdir, "pnb_legacy_summary.csv"))

cat("\nRun complete. Results saved to results/pnb_legacy_summary.csv\n")
