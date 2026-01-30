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

cat("After filtering:", nrow(simdata), "observations\n")
cat("Margin range:", range(simdata$margin), "\n")
cat("Rate range:", range(simdata$rate_deposits), "\n")

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
cat("Data summary saved.\n")

# Aggregates for Covariates
total_market_deposits <- sum(simdata$total_deposits, na.rm = TRUE)
bank_assets <- simdata$total_assets

# Stan Data Preparation
sdata_template <- list(
  use_cutoff = 0L,
  N = nrow(simdata),
  shareIn = simdata$shareIn,
  marginInv = simdata$marginInv,
  rateDiff = as.numeric(scale(simdata$rate_deposits)),
  loan_rate = rep(0, nrow(simdata)),
  
  N_event = 1, 
  event = rep(1L, nrow(simdata)),
  
  N_tophold = nlevels(simdata$tophold), 
  tophold = as.integer(simdata$tophold),
  
  # Covariates
  log_deposits = as.array(0.0),  # Single market, no between-market variation
  log_assets = as.numeric(scale(log(bank_assets))),
  
  rateDiff_sd = sd(simdata$rate_deposits),
  
  N_year = 1, 
  year = rep(1L, nrow(simdata)),
  
  N_market_year = 1, 
  market_year_idx = rep(1L, nrow(simdata)),
  
  grainsize = 1,
  
  # --- SINGLE MARKET MODE ---
  is_single_market = 1L,
  use_hmt = 0L,
  avg_price_hmt = mean(simdata$rate_deposits),
  avg_margin_hmt = mean(simdata$margin),
  ssnip_hmt = 0.05,
  
  # --- PRIOR SCALES (Data Driven) ---
  prior_sigma_share = sd(log(simdata$shareIn)), 
  prior_sigma_margin = sd(simdata$marginInv)
)

# Compile Model
cat("Compiling Stan Model...\n")
stan_mod <- stan_model(modelpath)

# --- RUN MODELS ---
models <- c("Bertrand", "Auction", "Cournot", "MonCom")
results <- list()

for (m in 1:4) {
  model_name <- models[m]
  cat(sprintf("\n=== RUNNING MODEL: %s ===\n", model_name))
  
  sdata <- sdata_template
  sdata$supply_model <- as.integer(m)
  
  # Lower adapt_delta for speed (data has variance now)
  adapt_delta <- 0.90
  if (m == 3) adapt_delta <- 0.95  # Cournot needs slightly more
  
  # Sampling
  fit <- sampling(
    stan_mod,
    data = sdata,
    chains = CHAINS,
    cores = CORES,
    iter = ITER + WARMUP,
    warmup = WARMUP,
    seed = SEED,
    control = list(adapt_delta = adapt_delta, max_treedepth = 12)
  )
  
  # Check Divergences
  sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
  divs <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
  cat(sprintf("Divergences: %d\n", divs))
  
  # Bridge Sampling (Marginal Likelihood)
  cat("Computing Bridge Sampling LogML...\n")
  logml_res <- tryCatch({
    bridge_sampler(fit, silent = TRUE, maxiter = 5000)
  }, error = function(e) {
    cat("Bridge Error:", e$message, "\n")
    return(NULL)
  })
  
  logml <- if (!is.null(logml_res)) logml_res$logml else NA
  cat(sprintf("LogML: %.2f\n", logml))
  
  # Extract Parameters
  post <- rstan::extract(fit)
  
  # Calculate Posterior Means and SDs
  p_alpha_mean <- mean(post$a_event)
  p_alpha_sd   <- sd(post$a_event)
  
  # Convert S0 from Logit to Probability Scale
  p_s0_mean <- mean(plogis(post$s0))
  p_s0_sd   <- sd(plogis(post$s0))
  
  p_rho_mean <- mean(post$rho_gen)
  p_rho_sd   <- sd(post$rho_gen)
  
  p_sigmam_mean <- mean(post$sigma_margin)
  p_sigmam_sd   <- sd(post$sigma_margin)
  
  p_sigmas_mean <- mean(post$sigma_logshare)
  p_sigmas_sd   <- sd(post$sigma_logshare)
  
  # Store for Summary Table
  results[[model_name]] <- list(
    Model = model_name,
    Divergences = divs,
    LogML = logml,
    Alpha = p_alpha_mean,
    S0 = p_s0_mean,
    Rho = p_rho_mean,
    SigmaMargin = p_sigmam_mean,
    
    # Store full details for parameter table
    details = data.frame(
      Parameter = c("Alpha", "S0", "Sigma_Share", "Sigma_Margin", "Rho"),
      Mean = c(p_alpha_mean, p_s0_mean, p_sigmas_mean, p_sigmam_mean, p_rho_mean),
      SD = c(p_alpha_sd, p_s0_sd, p_sigmas_sd, p_sigmam_sd, p_rho_sd)
    )
  )
}

# --- MODEL COMPARISON TABLE ---
cat("\n=== GENERATING COMPARISON TABLE ===\n")
# Extract main results (excluding details)
df_res <- bind_rows(lapply(results, function(x) x[names(x) != "details"]))

max_logml <- max(df_res$LogML, na.rm = TRUE)
df_res <- df_res %>%
  mutate(
    diff_logml = LogML - max_logml,
    bayes_factor_vs_best = exp(diff_logml),
    posterior_prob = exp(diff_logml) / sum(exp(diff_logml), na.rm = TRUE)
  ) %>%
  arrange(desc(LogML)) %>%
  select(Model, LogML, posterior_prob, bayes_factor_vs_best, Divergences, Alpha, S0, Rho, SigmaMargin)

print(as.data.frame(df_res))

# Save Results
outfile <- file.path(resultsdir, "pnb_model_results.csv")
write_csv(df_res, outfile)
cat(sprintf("\nResults saved to: %s\n", outfile))

# --- GENERATE PARAMETER COMPARISON TABLE ---
cat("\n=== GENERATING PARAMETER COMPARISON TABLE ===\n")

# Prior Definitions (Approximate based on Stan code)
# Alpha ~ LogNormal(0, 1) -> Mean ≈ 1.6, SD ≈ 2.1 (Very rough approx for transformed param)
# S0 ~ Logistic(-1, 0.5) -> Mean ≈ 0.27
# Sigma ~ HalfNormal -> Mean depends on truncation
priors <- data.frame(
  Parameter = c("Alpha", "S0", "Sigma_Share", "Sigma_Margin", "Rho"),
  Prior_Desc = c("LogNorm(0,0.5)", "Logit(-0.7,1)", "Data Scaled", "Data Scaled", "LKJ(4)"),
  Prior_Mean = c("1.13", "0.33", "Data SD", "Data SD", "0.00"),
  Prior_SD   = c("0.60", "0.20", "Data SD", "Data SD", "0.25") 
)

# Format Prior Column
priors$Prior_Col <- paste0(priors$Prior_Mean, " (", priors$Prior_SD, ")")

# Combine Models
param_table <- priors %>% select(Parameter, Prior = Prior_Col)

for (m in models) {
  det <- results[[m]]$details
  det$Formatted <- sprintf("%.3f (%.3f)", det$Mean, det$SD)
  colnames(det)[colnames(det) == "Formatted"] <- m
  
  param_table <- left_join(param_table, det %>% select(Parameter, all_of(m)), by = "Parameter")
}

# Print and Save
print(param_table)
param_outfile <- file.path(resultsdir, "pnb_parameter_comparison.csv")
write_csv(param_table, param_outfile)
cat(sprintf("\nParameter table saved to: %s\n", param_outfile))
