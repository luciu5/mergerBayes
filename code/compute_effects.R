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
  
  n_total <- length(draws$lp__)
  set.seed(42)
  idx <- sample(seq_len(n_total), min(N_DRAWS, n_total))

  prices <- as.numeric(edata$rate_deposits)
  shares_cond <- as.numeric(edata$shareIn)
  ownerPre <- ownerPost <- edata$tophold
  ownerPost[target_idx] <- ownerPost[acquirer_idx]

if(all.equal(ownerPre, ownerPost)) {
  stop("OwnerPre and ownerPost are equal\n")
}

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


# 2. Bayesian approach (from fit files)
models <- c("Bertrand", "Auction", "Cournot", "MonCom")

for (model_name in models) {
  # Update filename pattern for PNB 1960 run
  fit_file <- file.path(resultsdir, paste0("pnb_fit_", model_name, ".rds"))
  
  if (!file.exists(fit_file)) {
    cat("Skipping", model_name, "- no fit file\n")
    next
  }

  cat("Loading fit:", model_name, "\n")
  fit <- readRDS(fit_file)
  draws <- rstan::extract(fit)
  
  # Determine Supply/Demand mode for antitrust package
  # Bertrand -> supply="bertrand", demand="logit"
  # Auction -> supply="auction", demand="logit"
  # Cournot -> supply="cournot", demand="logit" (or linear?) - Standard Cournot implies quantities
  # MonCom -> supply="bertrand"? (Monopolistic Competition is technically Bertrand with potentially different elasticities, but standard antitrust package doesn't have "moncom". treat as Bertrand/Logit)
  
  supply_mode <- "bertrand"
  if (model_name == "Auction") supply_mode <- "auction"
  if (model_name == "Cournot") supply_mode <- "cournot"
  
  for (e in seq_along(event_levels)) {
    eid <- event_levels[e]
    # Filter for Event 1 (PNB)
    if (eid != "1") next 
    
    edata <- simdata %>% filter(event == eid)
    cat(sprintf("  %s Event %s (%d firms): ", model_name, eid, nrow(edata)))
    
    # Custom simulation wrapper
    price_changes <- vapply(seq_len(min(N_DRAWS, length(draws$lp__))), function(i) {
        alpha_raw <- draws$mu_log_a[i] 
        alpha <- exp(alpha_raw) / rate_sd
        
        # S0: draws$s0_logit is [Iter, N_market]. 
        s0_logit <- draws$s0_logit[i, 1] 
        s0 <- plogis(s0_logit)
        
        # Prepare Antitrust sim
        shares_cond <- as.numeric(edata$shareIn)
        prices <- as.numeric(edata$rate_deposits)
        
        # Determine ID column
        id_col <- "id_rssd"
        if (!("id_rssd" %in% names(edata))) {
             if ("rssdid" %in% names(edata)) id_col <- "rssdid"
             else if ("idrssd" %in% names(edata)) id_col <- "idrssd"
             else id_col <- "tophold"
        }
        
        # Get numeric IDs
        firm_ids <- as.numeric(as.character(edata[[id_col]]))
        
        # Filter events table for current event to get target/acquirer IDs
        # Note: 'events' table loaded globally
        ev <- events %>% filter(event_id == as.integer(eid))
        
        t_idx <- which(firm_ids == ev$target)
        a_idx <- which(firm_ids == ev$acquirer)
        
        ownerPre <- as.character(edata[[id_col]])
        ownerPost <- ownerPre
        if(length(t_idx) > 0 && length(a_idx) > 0) {
            # Assign Acquirer ID to Target (Merge)
            acq_label <- ownerPost[a_idx[1]]
            ownerPost[t_idx] <- acq_label
        } else {
             # If target/acquirer not found in this event data (e.g. dropped due to share < threshold), return 0
             return(0.0)
        }
        
        # Check if ownerPost differs
        if (all(ownerPost == ownerPre)) return(0.0)
        
        # Monopolistic Competition theoretical result: No strategic effect from merger
        if (model_name == "MonCom") return(0.0)

        res_sim <- tryCatch({
            firm_labels <- as.character(edata[[id_col]])
            
            # Unconditional shares for meanval calculation
            shares_uncond <- shares_cond * (1 - s0)
            meanval <- log(shares_uncond) - log(s0) - alpha * prices
                
            # Use sim() for all models with consistent interface
            sim_supply <- if (model_name == "Auction") "auction" else 
                          if (model_name == "Cournot") "cournot" else 
                          "bertrand" # Default (Bertrand/MonCom)
            
            antitrust::sim(prices, supply=sim_supply, demand="Logit", 
                 demand.param=list(alpha=alpha, meanval=meanval),
                 ownerPre=ownerPre, ownerPost=ownerPost, labels=firm_labels)

        }, error = function(e) {
             if (i == 1) cat(paste("    [ERROR] Simulation failed:", e$message, "\n"))
             NULL
        })
        
        if (is.null(res_sim)) return(NA)
        
        # Calculate Price Effect
        # IMPORTANT: pre=FALSE to get Post-Merger prices
        pppost <- antitrust::calcPrices(res_sim, pre=FALSE)
        chg <- (pppost / prices) - 1
        
        # Return Merging Parties Avg Change (weighted by pre-merger share)
        if(length(t_idx) > 0 && length(a_idx) > 0) {
            party_indices <- c(t_idx, a_idx)
            sum(chg[party_indices] * shares_cond[party_indices]) / sum(shares_cond[party_indices])
        } else {
            sum(chg * shares_cond)
        }
    }, numeric(1))

    valid <- price_changes[!is.na(price_changes)]
    
    if (length(valid) > 0) {
      cat(sprintf("%.4f (%.4f) [%d/%d valid]\n", mean(valid), sd(valid), length(valid), N_DRAWS))
      
      all_results[[length(all_results) + 1]] <- data.frame(
        Event = as.integer(eid),
        Method = paste0("Bayesian_", model_name),
        N_Firms = nrow(edata),
        Mean_Price_Change = mean(valid),
        SD_Price_Change = sd(valid),
        N_Valid = length(valid)
      )
    }
  }
}

# 2. Calibration approach (iterating over models)
cat("\n--- Calibration (ALM) ---\n")
cal_models <- c("Bertrand", "Cournot", "Auction", "MonCom")

for (e in seq_along(event_levels)) {
  eid <- event_levels[e]
  edata <- simdata %>% filter(event == eid)
  # Only calibrate for PNB (Event 1) as requested table focus, or all? 
  # Let's do all events just in case.
  
  cat(sprintf("  Calibration Event %s (%d firms):\n", eid, nrow(edata)))
  
  # Determine ID column
  id_col <- "id_rssd"
  if (!("id_rssd" %in% names(edata))) {
       if ("rssdid" %in% names(edata)) id_col <- "rssdid"
       else if ("idrssd" %in% names(edata)) id_col <- "idrssd"
       else id_col <- "tophold"
  }
  
  firm_ids <- as.numeric(as.character(edata[[id_col]]))
  ev_info <- events %>% filter(event_id == as.integer(eid))
  t_idx <- which(firm_ids == ev_info$target)
  a_idx <- which(firm_ids == ev_info$acquirer)
  
  ownerPre <- diag(nrow(edata)); 
  # Better: use labels for owner matrix if possible, but alm functions usually take matrix
  # If we have identifiers, we can build the matrix.
  # Simplest: Pre-merger is diagonal (competitors)
  # Post-merger: Target and Acquirer share ownership (1s in consistent columns)
  
  ownerPost <- ownerPre
  if(length(t_idx) > 0 && length(a_idx) > 0) {
      ownerPost[t_idx, a_idx[1]] <- 1 # Target now owned by Acquirer
      ownerPost[a_idx[1], t_idx[1]] <- 1 # Acquirer now owns Target (if matrix is symmetric/common ownership)
      # Usually strict ownership: Target row -> Acquirer Col = 1?
      # Antitrust package conventions: 
      # ownerPre: matrix where (i,j)=1 if common ownership.
      # Merger: set (target, acquirer) = 1 and (acquirer, target) = 1
      ownerPost[t_idx, a_idx] <- 1
      ownerPost[a_idx, t_idx] <- 1
  }

  for (model_name in cal_models) {
      if (model_name == "MonCom") {
          price_change <- 0.0
          cal_alpha <- NA
          cal_s0 <- NA
          cat(sprintf("    %s: 0.0000 (Theoretical)\n", model_name))
      } else {
          res_cal <- tryCatch({
              if (model_name == "Bertrand") {
                  antitrust::bertrand.alm(
                      demand = "logit",
                      prices = as.numeric(edata$rate_deposits),
                      quantities = as.numeric(edata$shareIn),
                      margins = as.numeric(edata$margin),
                      ownerPre = ownerPre,
                      ownerPost = ownerPost,
                      labels = as.character(edata[[id_col]]),
                      output = FALSE
                  )
              } else if (model_name == "Cournot") {
                  antitrust::logit.cournot.alm(
                      prices = as.numeric(edata$rate_deposits),
                      shares = as.numeric(edata$shareIn),
                      margins = as.numeric(edata$margin),
                      ownerPre = ownerPre,
                      ownerPost = ownerPost,
                      labels = as.character(edata[[id_col]]),
                      output = FALSE
                  )
              } else if (model_name == "Auction") {
                  antitrust::auction2nd.logit.alm(
                      prices = as.numeric(edata$rate_deposits),
                      shares = as.numeric(edata$shareIn),
                      margins = as.numeric(edata$margin),
                      ownerPre = ownerPre,
                      ownerPost = ownerPost,
                      labels = as.character(edata[[id_col]]),
                      output = FALSE
                  )
              }
          }, error = function(e) {
              cat(sprintf("    %s Error: %s\n", model_name, e$message))
              NULL
          })
          
          if (!is.null(res_cal)) {
               # Calc Price Change
               p_post <- antitrust::calcPrices(res_cal, pre=FALSE)
               p_pre <- as.numeric(edata$rate_deposits) 
               
               chg <- (p_post / p_pre) - 1
               
               # Weighted average for merging parties
               if(length(t_idx) > 0 && length(a_idx) > 0) {
                   party_idx <- c(t_idx, a_idx)
                   w <- as.numeric(edata$shareIn)[party_idx]
                   price_change <- sum(chg[party_idx] * w) / sum(w)
               } else {
                   price_change <- sum(chg * as.numeric(edata$shareIn))
               }
               
               # Note: For deposit markets, a standard product merger simulation (bertrand.alm) 
               # predicts a positive price effect (rate increase). 
               # However, market power in deposits typically leads to rate DECREASES.
               # We report the raw calculated change here.
               
               cal_alpha <- tryCatch(mean(abs(res_cal@slopes$alpha)), error=function(e) NA) 
               
               cat(sprintf("    %s: %.4f\n", model_name, price_change))
          } else {
               price_change <- NA
               cal_alpha <- NA
          }
      }

      all_results[[length(all_results) + 1]] <- data.frame(
        Event = as.integer(eid),
        Method = paste0("Calibration_", model_name),
        N_Firms = nrow(edata),
        Mean_Price_Change = price_change,
        SD_Price_Change = NA,
        N_Valid = 1
      )
  }
}

# ---- Save results ----
effects_summary <- bind_rows(all_results)
cat("\n=== Results ===\n")
print(effects_summary)
write_csv(effects_summary, file.path(resultsdir, "pnb_effects_summary.csv"))
cat("\nSaved:", file.path(resultsdir, "pnb_effects_summary.csv"), "\n")
