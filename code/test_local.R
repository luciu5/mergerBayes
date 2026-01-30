# ---- Local Test Script ----
# Quick test with 5 markets to verify Stan model compiles and runs
rm(list = ls())

library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(rstan)

cat("=== LOCAL TEST: 5 markets, minimal iterations ===\n")

# Use 2 cores for chains, 1 thread per chain (safe for local testing)
options(mc.cores = 2)
rstan_options(auto_write = TRUE)
rstan_options(threads_per_chain = 1)

# Paths
directory <- "d:/Projects/mergerBayes"
datadir <- file.path(directory, "data")
resultsdir <- file.path(directory, "results")
modelpath <- file.path(directory, "code", "bayesian.stan")

# Ensure results directory exists
if (!dir.exists(resultsdir)) {
    dir.create(resultsdir, recursive = TRUE)
}

# Load data
cat("Loading data...\n")
load(file = file.path(datadir, "banksimdata.RData"))

# Process data (same as main script)
simdata <- simdata %>%
    filter(year >= 2014 & year <= 2015) %>%
    filter(deposit_share > 0) %>%
    mutate(
        event_mkt = interaction(event_id, fedmkt, drop = TRUE),
        tophold = factor(tophold),
        year = factor(year),
        shareIn = deposit_share,
        margin = selected_margin / 100,
        rate_deposits = rate_call_report / 100,
        rate_loans = 0
    ) %>%
    mutate(margin = ifelse(margin <= 0, NA, margin)) %>%
    filter(!is.na(margin)) %>%
    mutate(
        event_mkt = droplevels(event_mkt),
        tophold = droplevels(tophold),
        year = droplevels(year)
    )

# *** SUBSAMPLE: Take only 5 unique markets ***
set.seed(42) # For reproducibility
sample_markets <- sample(levels(simdata$event_mkt), min(5, nlevels(simdata$event_mkt)))
simdata <- simdata %>%
    filter(event_mkt %in% sample_markets) %>%
    mutate(
        event_mkt = droplevels(event_mkt),
        tophold = droplevels(tophold),
        year = droplevels(year)
    ) %>%
    arrange(event_mkt, year, tophold)

cat("Subsample size:", nrow(simdata), "observations\n")
cat("N_event (markets):", nlevels(simdata$event_mkt), "\n")
cat("N_tophold (banks):", nlevels(simdata$tophold), "\n")
cat("N_year:", nlevels(simdata$year), "\n")

if (nrow(simdata) < 10) {
    stop("Too few observations after filtering!")
}

# Market-level summary
eventdata <- simdata %>%
    group_by(event_mkt) %>%
    summarise(deposit_total_market = mean(deposit_total_market, na.rm = TRUE)) %>%
    arrange(event_mkt)

# Bank-level summary
topholddata <- simdata %>%
    group_by(tophold) %>%
    summarise(total_deposits = mean(deposits, na.rm = TRUE)) %>%
    arrange(tophold)

# market-year indices
simdata <- simdata %>%
    mutate(market_year = interaction(event_mkt, year, drop = TRUE))

N_market_year <- nlevels(simdata$market_year)
simdata <- simdata %>%
    mutate(market_year_idx = as.numeric(market_year))

# Prepare Stan data
stan_data <- list(
    supply_model = 1L, # Bertrand
    use_cutoff = 0L,
    N = nrow(simdata),
    shareIn = simdata$shareIn,
    marginInv = 1 / simdata$margin,
    rateDiff = as.numeric(scale(simdata$rate_deposits, center = TRUE, scale = TRUE)),
    loan_rate = rep(0, nrow(simdata)),
    N_event = nlevels(simdata$event_mkt),
    event = as.integer(simdata$event_mkt),
    N_tophold = nlevels(simdata$tophold),
    tophold = as.integer(simdata$tophold),
    log_deposits = as.numeric(scale(log(eventdata$deposit_total_market), center = TRUE, scale = TRUE)),
    log_assets = as.numeric(scale(log(topholddata$total_deposits), center = TRUE, scale = TRUE)),
    rateDiff_sd = sd(simdata$rate_deposits, na.rm = TRUE),
    N_year = nlevels(simdata$year),
    year = as.numeric(simdata$year),
    N_market_year = N_market_year,
    market_year_idx = as.integer(simdata$market_year_idx),
    grainsize = 1 # Small grainsize for test
)

cat("\n=== Compiling Stan model... ===\n")
model <- stan_model(modelpath)
cat("Compilation successful!\n")

cat("\n=== Running short sampling (2 chains, 200 iter)... ===\n")
system.time(fit <- sampling(
    model,
    data = stan_data,
    chains = 2,
    iter = 200,
    warmup = 100,
    thin = 1,
    init = 0,
    control = list(adapt_delta = 0.9, max_treedepth = 10)
))

cat("\n=== Sampling complete! ===\n")

# Quick diagnostic
cat("\nKey parameter summary:\n")
print(summary(fit, pars = c("sigma_logshare", "sigma_margin", "rho_gen"))$summary)

# Check for issues
sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
divergences <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
cat("\nDivergent transitions:", divergences, "\n")

# Save minimal test results
test_outfile <- file.path(resultsdir, "test_local_result.RData")
save(fit, stan_data, file = test_outfile, compress = "xz")
cat("\nTest results saved to:", test_outfile, "\n")

cat("\n=== TEST PASSED ===\n")
