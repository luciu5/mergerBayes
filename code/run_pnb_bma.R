# ---- PNB Model Comparison & BMA (Decoupled, Tight Margins) ----
# Runs all 4 models fitting margins with prior_sigma_margin=0.2.
# Forces Rho=0 (Decoupled).

library(dplyr)
library(tidyr)
library(rstan)
library(readr)
library(loo)

# --- CONFIGURATION ---
ITER <- 2000
WARMUP <- 1000
CHAINS <- 4
CORES <- 4
SEED <- 42

directory <- "d:/Projects/mergerBayes"
datadir <- file.path(directory, "data")
resultsdir <- file.path(directory, "results")
modelpath <- file.path(directory, "code", "bayesian.stan")

# Load Data
load(file.path(datadir, "supreme_data.RData")) 
models <- c("Bertrand", "Auction", "Cournot", "MonComp")

# Compile Model
stan_mod <- stan_model(modelpath)

# Initialize Output File
outfile <- file.path(resultsdir, "pnb_bma_results_informative_alpha.csv")
cat("Model,Observations,LOOIC,Alpha_Mean,Alpha_SD,Cutoff_Mean,Cutoff_SD,S0_Mean,S0_SD,SigmaMargin_Mean,SigmaMargin_SD,Rho_Mean,Rho_SD\n", file = outfile)

# --- PREP DATA (Event 1) ---
eid <- 1
evt_data <- simdata %>% 
  filter(event_id == eid) %>%
  mutate(
    marginInv = 1 / margin,
    rateDiff = rate_deposits, 
    tophold = as.integer(factor(tophold)),
    year = 1L
  ) %>%
  filter(is.finite(marginInv) & margin > 0)

sdata <- list(
  N = nrow(evt_data),
  shareIn = evt_data$shareIn,
  marginInv = evt_data$marginInv,
  rateDiff = as.numeric(scale(evt_data$rate_deposits)),
  rateDiff_sd = sd(evt_data$rate_deposits),
  
  N_event = 1,
  event = rep(1L, nrow(evt_data)),
  events = as.array(1L), 
  N_tophold = max(evt_data$tophold),
  tophold = evt_data$tophold,
  log_deposits = as.array(0.0),
  N_year = 1,
  year = rep(1L, nrow(evt_data)),
  grainsize = 1, 
  is_single_market = 1L, # Forces Rho=0 in transformed parameters (Decoupled)
  use_hmt = 0L,
  fix_supply_intercept = 1L,
  min_s0 = as.array(0.0), 
  avg_price_hmt = 0, avg_margin_hmt = 0, ssnip_hmt = 0.05,
  
  # PRIORS (Balanced - Alpha Prior Does the Heavy Lifting)
  prior_sigma_share = 0.01,          # 1% share tolerance
  prior_sigma_margin = 1.0,          # Moderate margin tolerance (scaled for inverse)
  prior_sigma_meanval_strat = 2.0,   # Allow strategic heterogeneity
  prior_sigma_meanval_fringe = 0.5,  # Allow fringe variation
  prior_lkj = 4.0                    # (Unused with is_single_market=1)
)

# --- LOOP MODELS ---
for (m in 1:4) {
  model_name <- models[m]
  
  # Logic: MonComp (4) -> No Cutoff
  cutoff_flag <- if (m == 4) 0L else 1L
  sdata$use_cutoff <- cutoff_flag
  sdata$supply_model <- m
  
  cat(sprintf("Fitting %s (Cutoff=%d, Informative Alpha Prior)...\n", model_name, cutoff_flag))
  
  # Refresh=100 so log shows progress
  fit <- sampling(
    stan_mod, data = sdata, chains = CHAINS, cores = CORES,
    iter = ITER + WARMUP, warmup = WARMUP, seed = SEED,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 100 
  )
  
  # Extract Parameters
  loo_val <- NA
  try({
     loo_res <- loo(fit)
     loo_val <- loo_res$estimates["looic", "Estimate"]
  }, silent=TRUE)
  
  # Alpha (Unscaled)
  a_event_s <- extract(fit, "a_event")[[1]]
  alpha_mean <- mean(a_event_s) / sdata$rateDiff_sd
  alpha_sd   <- sd(a_event_s)   / sdata$rateDiff_sd
  
  # Cutoff
  if (cutoff_flag == 1) {
    cut_s <- extract(fit, "cutoff_share")[[1]]
    cut_mean <- mean(cut_s)
    cut_sd   <- sd(cut_s)
  } else {
    cut_mean <- NA 
    cut_sd   <- NA
  }
  
  # S0
  s0_s <- extract(fit, "s0")[[1]]
  s0_prob <- plogis(s0_s)
  s0_mean <- mean(s0_prob)
  s0_sd   <- sd(s0_prob)

  # Sigma Margin
  sm_s <- extract(fit, "sigma_margin")[[1]]
  sm_mean <- mean(sm_s)
  sm_sd   <- sd(sm_s)
  
  # Rho (Should be 0 distribution)
  rescor_s <- extract(fit, "Rescor")[[1]]
  rho_vals <- rescor_s[, 1, 2] 
  rho_mean <- mean(rho_vals)
  rho_sd   <- sd(rho_vals)
  
  # Save
  row_df <- data.frame(
    Model = model_name,
    Observations = nrow(evt_data),
    LOOIC = round(loo_val, 2),
    Alpha_Mean = alpha_mean,
    Alpha_SD = alpha_sd,
    Cutoff_Mean = cut_mean,
    Cutoff_SD = cut_sd,
    S0_Mean = s0_mean,
    S0_SD = s0_sd,
    SigmaMargin_Mean = sm_mean,
    SigmaMargin_SD = sm_sd,
    Rho_Mean = rho_mean,
    Rho_SD = rho_sd
  )
  
  write.table(row_df, outfile, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, quote = FALSE)
  cat("Saved.\n")
}
