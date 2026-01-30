# ---- Local Test Script: All 4 Models + Bridge Sampling ----
rm(list = ls())

library(dplyr)
library(tidyr)
library(rstan)
library(bridgesampling)
library(loo)

# Configuration
TEST_ITER <- 400
TEST_WARMUP <- 200
MODELS_TO_TEST <- 1:4 # 1=Bertrand, 2=Auction, 3=Cournot, 4=MonCom
MODEL_NAMES <- c("Bertrand", "Auction", "Cournot", "MonCom")

cat("=== ROBUST LOCAL TEST: All 4 Models + Bridge Sampling ===\n")

# Use minimal cores (1 chain per model test to save time, or 2 if fast)
options(mc.cores = 2)
rstan_options(auto_write = TRUE)

# Paths
directory <- "d:/Projects/mergerBayes"
datadir <- file.path(directory, "data")
modelpath <- file.path(directory, "code", "bayesian.stan")

# Load and Prep Data Once
cat("Loading and prepping 5-market subsample...\n")
load(file = file.path(datadir, "banksimdata.RData"))

# Filter & Process
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
        loan_rate = 0
    ) %>%
    mutate(margin = ifelse(margin <= 0, NA, margin)) %>%
    filter(!is.na(margin)) %>%
    mutate(
        event_mkt = droplevels(event_mkt),
        tophold = droplevels(tophold),
        year = droplevels(year)
    )

# Filter Small Markets (Sync with bayesian.R)
mkt_counts <- simdata %>%
    group_by(event_mkt, year) %>%
    summarise(n_firms = n(), .groups = "drop")
bad_markets <- mkt_counts %>%
    filter(n_firms <= 2) %>%
    pull(event_mkt) %>%
    unique()
if (length(bad_markets) > 0) {
    cat(paste("Dropping", length(bad_markets), "markets with <= 2 firms.\n"))
    simdata <- simdata %>% filter(!event_mkt %in% bad_markets)
}

# Subsample 5 markets
set.seed(42)
sample_markets <- sample(levels(simdata$event_mkt), 5)
simdata <- simdata %>%
    filter(event_mkt %in% sample_markets) %>%
    mutate(
        event_mkt = droplevels(event_mkt),
        tophold = droplevels(tophold),
        year = droplevels(year)
    ) %>%
    arrange(event_mkt, year, tophold)

# Helper Data
eventdata <- simdata %>%
    group_by(event_mkt) %>%
    summarise(deposit_total_market = mean(deposit_total_market))
topholddata <- simdata %>%
    group_by(tophold) %>%
    summarise(total_deposits = mean(deposits))
simdata <- simdata %>% mutate(market_year = interaction(event_mkt, year, drop = TRUE), market_year_idx = as.numeric(market_year))
N_market_year <- nlevels(simdata$market_year)

cat("Subsample ready:", nrow(simdata), "obs.\n")

# Compile Model
cat("Compiling Stan model...\n")
stan_mod <- stan_model(modelpath)

# Loop through models
results_list <- list()

for (m in MODELS_TO_TEST) {
    cat("\n--------------------------------------------------------\n")
    cat("TESTING MODEL:", MODEL_NAMES[m], "\n")
    cat("--------------------------------------------------------\n")

    # Adapt Delta
    target_adapt <- 0.90
    if (m == 3) target_adapt <- 0.99 # Boost for Cournot

    # Stan Data
    sdata <- list(
        supply_model = as.integer(m), use_cutoff = 0L, N = nrow(simdata),
        shareIn = simdata$shareIn, marginInv = 1 / simdata$margin,
        rateDiff = as.numeric(scale(simdata$rate_deposits)), loan_rate = rep(0, nrow(simdata)),
        N_event = nlevels(simdata$event_mkt), event = as.integer(simdata$event_mkt),
        N_tophold = nlevels(simdata$tophold), tophold = as.integer(simdata$tophold),
        log_deposits = as.numeric(scale(log(eventdata$deposit_total_market))),
        log_assets = as.numeric(scale(log(topholddata$total_deposits))),
        rateDiff_sd = sd(simdata$rate_deposits),
        N_year = nlevels(simdata$year), year = as.numeric(simdata$year),
        N_market_year = N_market_year, market_year_idx = as.integer(simdata$market_year_idx),
        grainsize = 1,

        # --- NEW FLAGS (Default to 0 for Test) ---
        is_single_market = 0L,
        use_hmt = 0L,
        avg_price_hmt = 0.0,
        avg_margin_hmt = 0.0,
        ssnip_hmt = 0.05
    )

    # Sampling
    fit <- sampling(stan_mod,
        data = sdata, chains = 2, iter = TEST_ITER + TEST_WARMUP, warmup = TEST_WARMUP,
        thin = 1, init = 0, control = list(adapt_delta = target_adapt, max_treedepth = 10)
    )

    # Check Diagnostics
    sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
    divs <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))

    # Bridge Sampling
    cat("  -> Attempting Bridge Sampling... ")
    bridge_res <- tryCatch(
        {
            bridge_sampler(fit, silent = TRUE, maxiter = 1000)
        },
        error = function(e) {
            return(NULL)
        }
    )

    bridge_status <- if (is.null(bridge_res)) "FAILED" else paste("SUCCESS (LogML:", round(bridge_res$logml, 2), ")")
    cat(bridge_status, "\n")

    results_list[[MODEL_NAMES[m]]] <- list(
        Divergences = divs,
        Bridge = bridge_status
    )
}

cat("\n============================================\n")
cat("FINAL REPORT\n")
cat("============================================\n")
for (name in names(results_list)) {
    res <- results_list[[name]]
    cat(sprintf("%-10s | Divergences: %-3d | Bridge: %s\n", name, res$Divergences, res$Bridge))
}
cat("\nTest Complete.\n")
