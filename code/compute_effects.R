# ---- Merger Effects Simulation ----
# Loads Stan fits and computes predicted price effects using 'antitrust' package
# Two approaches compared:
#   1. Bayesian: use posterior draws of alpha and s0 from hierarchical model
#   2. Calibration: use bertrand.alm() which calibrates alpha from observed data

rm(list = ls())

library(dplyr)
library(tidyr)
library(rstan)
library(antitrust)
library(readr)

# Configuration
directory <- "d:/Projects/mergerBayes"
datadir <- file.path(directory, "data")
resultsdir <- file.path(directory, "results")
if (!dir.exists(resultsdir)) dir.create(resultsdir, recursive = TRUE)

N_DRAWS <- 200

# Load Data
load(file.path(datadir, "supreme_data.RData"))

# Load merger event info (target + acquirer per event)
load("d:/Projects/supreme_eff/data/event_overlaps.Rdata") # loads 'events', 'overlaps', 'ownership'
cat("Merger events loaded from event_overlaps.Rdata\n")

# Prepare data: same filtering as pnb.R
# Exclude Provident (3) and Houston (4)
simdata <- simdata %>%
  filter(!event_id %in% c(3, 4)) %>%
  droplevels() %>%
  mutate(
    event = factor(event_id),
    tophold = factor(tophold),
    marginInv = 1 / margin,
    loan_rate = 0
  ) %>%
  filter(is.finite(marginInv) & margin > 0) %>%
  filter(margin >= 0.1) %>%
  group_by(event) %>%
  filter(as.numeric(as.character(year)) == min(as.numeric(as.character(year)))) %>%
  ungroup() %>%
  mutate(year = factor(year)) %>%
  droplevels()

# Rescale shares to sum to 1 within each event-year (conditional shares)
simdata <- simdata %>%
  group_by(event, year) %>%
  mutate(shareIn = shareIn / sum(shareIn)) %>%
  ungroup()

# Calculate rate_sd for scaling alpha (consistent with Stan model)
rate_sd <- sd(simdata$rate_deposits)
cat("Rate SD (for alpha scaling):", rate_sd, "\n")

event_levels <- levels(simdata$event)
cat("Events:", paste(event_levels, collapse = ", "), "\n")

# ---- Bayesian simulation for one event ----
simulate_bayesian <- function(edata, draws, event_idx, mky_idx, eid, rate_sd) {
  # Filter events table for current event to get target/acquirer IDs
  ev <- events %>% filter(event_id == as.integer(eid))
  if (nrow(ev) != 1) stop(paste("Event info not found for ID:", eid))
  
  # Map IDs (rssdid) to indices in the current edata partition
  # Assuming edata$tophold corresponds to rssdid
  target_id <- ev$target   
  acquirer_id <- ev$acquirer
  
  target_idx <- which(as.numeric(as.character(edata$tophold)) == target_id)
  acquirer_idx <- which(as.numeric(as.character(edata$tophold)) == acquirer_id)
  
  # Construct Ownership Matrices
  n_firms <- nrow(edata)
  ownerPre <- diag(n_firms)
  ownerPost <- ownerPre
  
  # Merge if both parties are present in this market simulation
  if (length(target_idx) == 1 && length(acquirer_idx) == 1) {
    ownerPost[target_idx, acquirer_idx] <- 1
    ownerPost[acquirer_idx, target_idx] <- 1
  } else {
    # If one party was filtered out (e.g. < 0.1% share), the merger has no effect in this model
    # We proceed with ownerPost = ownerPre (no price change expected from ownership)
    # But alpha/s0 draws might still imply different elasticities than calibration.
    missing_party <- c()
    if (length(target_idx) == 0) missing_party <- c(missing_party, paste("Target", target_id))
    if (length(acquirer_idx) == 0) missing_party <- c(missing_party, paste("Acquirer", acquirer_id))
    cat(sprintf("[Note: %s filtered out] ", paste(missing_party, collapse=", ")))
  }

  n_total <- length(draws$lp__)
  set.seed(42)
  idx <- sample(seq_len(n_total), min(N_DRAWS, n_total))

  prices <- as.numeric(edata$rate_deposits)
  shares_cond <- as.numeric(edata$shareIn)

  first_error <- NULL
  results <- vapply(idx, function(i) {
    alpha_raw <- draws$a_event[i, event_idx]
    # Stan alpha is on standardized prices, so scale it to raw prices
    alpha <- alpha_raw / rate_sd
    
    s0 <- plogis(draws$s0_logit[i, mky_idx])

    # Unconditional shares
    shares_uncond <- shares_cond * (1 - s0)

    # Supply-side Utility: U = delta + alpha * P  (alpha > 0)
    # Inversion: delta = log(shares_uncond) - log(s0) - alpha * prices
    meanval <- log(shares_uncond) - log(s0) - alpha * prices

    res <- tryCatch(
      sim(
        prices = prices,
        supply = "bertrand",
        demand = "Logit",
        output = FALSE, # Supply-side simulation (Deposits)
        demand.param = list(alpha = alpha, meanval = meanval), # Positive alpha
        ownerPre = ownerPre,
        ownerPost = ownerPost
      ),
      error = function(e) {
        if (is.null(first_error)) first_error <<- conditionMessage(e)
        NULL
      }
    )

    if (is.null(res)) return(NA_real_)
    chg <- (res@pricePost / res@pricePre) - 1
    sum(chg * shares_cond)
  }, numeric(1))

  if (!is.null(first_error) && sum(is.na(results)) > 0) {
    cat(sprintf("\n    [%d failures, first error: %s]\n    ", sum(is.na(results)), first_error))
  }

  results
}

# ---- Calibration simulation for one event ----
simulate_calibration <- function(edata, eid) {
  # Filter events table for current event to get target/acquirer IDs
  ev <- events %>% filter(event_id == as.integer(eid))
  target_id <- ev$target   
  acquirer_id <- ev$acquirer
  
  target_idx <- which(as.numeric(as.character(edata$tophold)) == target_id)
  acquirer_idx <- which(as.numeric(as.character(edata$tophold)) == acquirer_id)
  
  n_firms <- nrow(edata)
  ownerPre <- diag(n_firms)
  ownerPost <- ownerPre
  
  if (length(target_idx) == 1 && length(acquirer_idx) == 1) {
    ownerPost[target_idx, acquirer_idx] <- 1
    ownerPost[acquirer_idx, target_idx] <- 1
  }

  res <- tryCatch(
    bertrand.alm(
      demand = "logit",
      prices = as.numeric(edata$rate_deposits),
      quantities = as.numeric(edata$shareIn),
      margins = as.numeric(edata$margin),
      ownerPre = ownerPre,
      ownerPost = ownerPost,
      labels = as.character(edata$tophold),
      output = FALSE
    ),
    error = function(e) { cat("error:", conditionMessage(e), "\n"); NULL }
  )

  if (is.null(res)) return(list(price_change = NA, alpha = NA, s0 = NA))

  chg <- (res@pricePost / res@pricePre) - 1
  avg_chg <- sum(chg * as.numeric(edata$shareIn))

  cal_alpha <- abs(res@slopes$alpha)
  cal_s0 <- 1 - sum(res@shares)

  list(price_change = avg_chg, alpha = cal_alpha, s0 = cal_s0)
}

# ---- Main loop ----
all_results <- list()

# 1. Bayesian approach (from fit files)
models <- c("Bertrand")
for (model_name in models) {
  fit_file <- file.path(resultsdir, paste0("pnb_fit_", model_name, ".rds"))
  if (!file.exists(fit_file)) {
    cat("Skipping", model_name, "- no fit file\n")
    next
  }

  cat("Loading fit:", model_name, "\n")
  fit <- readRDS(fit_file)
  draws <- rstan::extract(fit)

  for (e in seq_along(event_levels)) {
    eid <- event_levels[e]
    edata <- simdata %>% filter(event == eid)

    cat(sprintf("  Bayesian Event %s (%d firms): ", eid, nrow(edata)))
    price_changes <- simulate_bayesian(edata, draws, e, e, eid, rate_sd)

    valid <- price_changes[!is.na(price_changes)]
    if (length(valid) > 0) {
      cat(sprintf("%.4f (%.4f) [%d/%d valid]\n", mean(valid), sd(valid), length(valid), N_DRAWS))
    } else {
      cat(sprintf("ALL FAILED [0/%d valid]\n", N_DRAWS))
    }

    all_results[[length(all_results) + 1]] <- data.frame(
      Event = as.integer(eid),
      Method = "Bayesian",
      N_Firms = nrow(edata),
      Mean_Price_Change = if (length(valid) > 0) mean(valid) else NA,
      SD_Price_Change = if (length(valid) > 1) sd(valid) else NA,
      N_Valid = length(valid)
    )
  }

  rm(fit, draws)
  gc()
}

# 2. Calibration approach (no fit needed)
cat("\n--- Calibration (bertrand.alm) ---\n")
for (e in seq_along(event_levels)) {
  eid <- event_levels[e]
  edata <- simdata %>% filter(event == eid)

  cat(sprintf("  Calibration Event %s (%d firms): ", eid, nrow(edata)))
  cal <- simulate_calibration(edata, eid)
  cat(sprintf("%.4f (alpha=%.3f, s0=%.3f)\n", cal$price_change, cal$alpha, cal$s0))

  all_results[[length(all_results) + 1]] <- data.frame(
    Event = as.integer(eid),
    Method = "Calibration",
    N_Firms = nrow(edata),
    Mean_Price_Change = cal$price_change,
    SD_Price_Change = NA,
    N_Valid = 1
  )
}

# ---- Save results ----
effects_summary <- bind_rows(all_results)
cat("\n=== Results ===\n")
print(effects_summary)
write_csv(effects_summary, file.path(resultsdir, "pnb_effects_summary.csv"))
cat("\nSaved:", file.path(resultsdir, "pnb_effects_summary.csv"), "\n")
