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

# Process command line arguments
args <- commandArgs(trailingOnly = TRUE)
thismodel <- as.numeric(args[1])
thismodel[is.na(thismodel)] <- 1 # run bertrand if no command line argument

# Get chain count from command line (2nd argument)
chain_count <- as.numeric(args[2])
if (is.na(chain_count)) chain_count <- 4 # default to 4 chain

# Get thread count from command line (3rd argument)
thread_count <- as.numeric(args[3])
if (is.na(thread_count)) thread_count <- 1 # default to 1 threads if not specified

# Set up Stan parallelization
options(mc.cores = chain_count) # Use 1 core per chain since we'll use threads within each chain
rstan_options(auto_write = TRUE)
Sys.setenv(STAN_NUM_THREADS = thread_count)

# Configure Stan to use multiple threads per chain
rstan_options(threads_per_chain = thread_count)

# add arg to toggle cutoff (1 = enable, 0 = disable)
use_cutoff <- as.numeric(args[4])
if (is.na(use_cutoff)) use_cutoff <- 0

model_name <- c("bertrand", "2nd", "cournot", "moncom")
datadir <- "data"
modelpath <- "code/bayesian.stan"
outfile <- file.path(datadir, paste0("stan_hhiperform_", model_name[thismodel], ".RData"))

# Load 2010-2020 bank merger data
load(file = file.path(datadir, "banksimdata.RData"))

# Filter for LAST YEAR ONLY (2020) - Production Run
simdata <- simdata %>% filter(year == 2020)

# Clean and process data for the Stan model
simdata <- simdata %>%
  filter(deposit_share > 0) %>%
  mutate(
    # Unique market identifier (event_id + fedmkt)
    event_mkt = interaction(event_id, fedmkt, drop = TRUE),
    tophold = factor(tophold),
    year = factor(year),
    # Map variables to names used in the original bayesian.R/stan
    shareIn = deposit_share,
    # Convert percentage margin (e.g. 5.5) to decimal (0.055)
    margin = selected_margin / 100,
    rate_deposits = rate_call_report / 100,
    # Set proxy for missing loan_rate (cost shifter)
    rate_loans = 0
  ) %>%
  mutate(margin = ifelse(margin <= 0, NA, margin)) %>%
  filter(!is.na(margin)) %>%
  mutate(
    event_mkt = droplevels(event_mkt),
    tophold = droplevels(tophold),
    year = droplevels(year)
  ) %>%
  arrange(event_mkt, year, tophold)

# Market-level summary for hierarchical priors
eventdata <- simdata %>%
  group_by(event_mkt) %>%
  summarise(
    deposit_total_market = mean(deposit_total_market, na.rm = TRUE)
  ) %>%
  arrange(event_mkt)

# Bank-level summary for hierarchical priors
topholddata <- simdata %>%
  group_by(tophold) %>%
  summarise(
    total_deposits = mean(deposits, na.rm = TRUE)
  ) %>%
  arrange(tophold)

# Add market-year indices for parallelization
simdata <- simdata %>%
  mutate(market_year = interaction(event_mkt, year, drop = TRUE))

market_years <- levels(simdata$market_year)
N_market_year <- length(market_years)

simdata <- simdata %>%
  mutate(market_year_idx = as.numeric(market_year))

# Check data dimensions
cat("Number of observations after filtering:", nrow(simdata), "\n")
cat("N_event (market units):", nlevels(simdata$event_mkt), "\n")
cat("N_tophold:", nlevels(simdata$tophold), "\n")
cat("N_year:", nlevels(simdata$year), "\n")

# Stop if too few observations
if (nrow(simdata) < 10) {
  stop("Too few observations after filtering! Check your data source.")
}

# Create stan_data with parallelization fields
stan_data <- list(
  supply_model = as.integer(thismodel), # 1=Bertrand, 2=SSA, 3=Cournot, 4=MonCom
  use_cutoff = as.integer(use_cutoff),
  N = nrow(simdata),
  shareIn = simdata$shareIn,
  marginInv = 1 / simdata$margin,
  rateDiff = as.numeric(scale(simdata$rate_deposits, center = TRUE, scale = TRUE)),
  # Set scale to 1 if all zeros to avoid division by zero
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
  grainsize = max(1, round(nrow(simdata) / (10 * thread_count)))
)

# Export data for debugging if needed
write_json(stan_data, file.path(datadir, "stan_data.json"), pretty = TRUE, auto_unbox = TRUE)

# Compile the model
model <- stan_model(modelpath)

# Sample from the posterior
system.time(fit <- sampling(
  model,
  data = stan_data,
  chains = chain_count,
  init = 0,
  iter = 7000,
  warmup = 1500,
  control = list(adapt_delta = 0.95, max_treedepth = 13)
))

# Build parameter lists for summaries
base_pars <- c(
  "a_event", "logit_mu_s0", "s0", "Rescor",
  "sigma_logshare", "sigma_margin", "gamma_loan"
)

if (stan_data$use_cutoff == 1) {
  plot_pars <- c(base_pars, "cutoff_share")
  sum_pars <- c(base_pars, "cutoff_share", "year_effect_demand", "year_effect_supply")
} else {
  plot_pars <- base_pars
  sum_pars <- c(base_pars, "year_effect_demand", "year_effect_supply")
}

# Generate summaries
plot_sum <- try(plot(fit, pars = plot_pars))
fit_sum <- try(summary(fit, pars = sum_pars)$summary)

# Bridge sampling for model comparison
bridge <- try(bridge_sampler(fit, cores = thread_count))

# Extract predicted margins for diagnostic
margin_samples <- rstan::extract(fit, "inv_markup_pred")[[1]]
neg_count <- sum(margin_samples < 0)
cat("Number of negative predicted margins:", neg_count, "\n")

# Save results
save(fit, bridge, plot_sum, fit_sum, file = outfile)
