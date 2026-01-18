# ---- Load Libraries ----
library(dplyr)
library(tidyr)
library(antitrust)
library(rstan)
library(jsonlite)
library(parallel)

# Configuration
datadir <- "data"
resdir <- "results"
if (!dir.exists(resdir)) dir.create(resdir, recursive = TRUE)

# Load merger data
load(file.path(datadir, "banksimdata.RData"))

# Filter for 2020 and specific 10 mergers (MATCHING BAYESIAN.R)
simdata <- simdata %>% filter(year == 2020)
target_events <- c(2117, 2086, 2096, 2069, 2178, 2144, 2164, 2139, 2173, 2047)
simdata <- simdata %>% filter(event_id %in% target_events)

# Clean and process data EXACTLY as in bayesian.R
simdata <- simdata %>%
    filter(deposit_share > 0) %>%
    mutate(
        # Unique market identifier (event_id + fedmkt)
        event_mkt = interaction(event_id, fedmkt, drop = TRUE),
        tophold = factor(tophold),
        year = factor(year),
        # Map variables to names used in bayesian.R/stan
        shareIn = deposit_share,
        margin = selected_margin / 100,
        rate_deposits = rate_call_report / 100
    ) %>%
    mutate(margin = ifelse(margin <= 0, NA, margin)) %>%
    filter(!is.na(margin)) %>%
    mutate(
        event_mkt = droplevels(event_mkt),
        tophold = droplevels(tophold),
        year = droplevels(year)
    ) %>%
    arrange(event_mkt, year, tophold)

# Calculate scaling stats (must match bayesian.R)
rate_mean <- mean(simdata$rate_deposits, na.rm = TRUE)
rate_sd <- sd(simdata$rate_deposits, na.rm = TRUE)

# Models to process for simulation
# Note: moncom excluded from simulation since it predicts zero price effects
models <- c("bertrand", "cournot", "2nd")

# Main processing loop
for (m_name in models) {
    cat("\nProcessing model:", m_name, "\n")
    outfile <- file.path(datadir, paste0("stan_hhiperform_", m_name, ".RData"))

    if (!file.exists(outfile)) {
        cat("  Skip: Result file not found.\n")
        next
    }

    # Load Stan fit
    cat("  Loading Stan results...\n")
    load(outfile) # loads 'fit'

    # Extract parameters
    draws <- rstan::extract(fit)
    n_draws <- length(draws$lp__)
    cat("  N draws found:", n_draws, "\n")

    # Market levels from simdata (must match Stan order)
    event_levels <- levels(simdata$event_mkt)
    n_events <- length(event_levels)

    # Determine subset of draws for effect distribution (e.g. 50 draws for speed)
    # Production might use more (e.g. 1000)
    set.seed(42)
    draw_subset <- sample(1:n_draws, min(50, n_draws))

    # Map supply type
    supply_type <- ifelse(m_name == "2nd", "auction", m_name)

    # List to store all results
    all_results <- list()

    cat("  Computing effects for each market and sampled draw...\n")

    # Loop over markets
    for (e_idx in seq_along(event_levels)) {
        e_name <- event_levels[e_idx]
        e_data <- simdata %>% filter(event_mkt == e_name)

        # Identify merging parties
        isParty <- e_data$isParty
        if (sum(isParty) < 2) next # Skip if not a merger

        # Ownership matrices
        ownerPre <- diag(nrow(e_data))
        ownerPost <- ownerPre
        party_indices <- which(isParty)
        ownerPost[party_indices[1], party_indices[2]] <- 1
        ownerPost[party_indices[2], party_indices[1]] <- 1

        # Bank-specific tophold indices
        tophold_indices <- as.integer(e_data$tophold)

        # Year index (assuming same year per market event)
        year_index <- as.integer(e_data$year[1])

        # Subset draws results for this market
        market_draw_results <- lapply(draw_subset, function(d_idx) {
            # Robust indexing helper
            get_val <- function(draw_arr, d, i) {
                if (is.matrix(draw_arr)) {
                    return(draw_arr[d, i])
                }
                # For vectors (like year_effect_demand when N_year=1), just use d
                return(draw_arr[d])
            }

            # price coefficient (rescaled)
            alpha_stan <- draws$a_event[d_idx, e_idx]
            alpha_sim <- alpha_stan / rate_sd

            # Fixed effects + s0 - alpha * mean_p / sd_p
            fe <- draws$b_event[d_idx, e_idx] +
                draws$b_tophold_scaled[d_idx, tophold_indices] +
                get_val(draws$year_effect_demand, d_idx, year_index) +
                draws$s0[d_idx, e_idx]

            meanval_sim <- fe - alpha_stan * rate_mean / rate_sd

            # Sim parameters
            dp <- list(
                alpha = alpha_sim,
                meanval = as.vector(meanval_sim)
            )

            # Run simulation
            # Note: auction models don't accept 'output' argument
            sim_args <- list(
                prices = e_data$rate_deposits,
                supply = supply_type,
                demand = "Logit",
                demand.param = dp,
                ownerPre = ownerPre,
                ownerPost = ownerPost,
                labels = as.character(e_data$tophold)
            )

            # Add output argument only for bertrand/cournot
            if (supply_type %in% c("bertrand", "cournot")) {
                sim_args$output <- FALSE
            }

            res <- try(do.call(sim, sim_args), silent = TRUE)

            if (inherits(res, "try-error")) {
                return(NULL)
            }

            # Extract values explicitly
            price_pre <- res@pricePre
            price_post <- res@pricePost
            shares_pre <- calcShares(res, preMerger = TRUE)

            # Industry price change (share-weighted)
            ind_change <- sum((price_post / price_pre - 1) * shares_pre) * 100

            # Merging party price change
            party_share_pre <- sum(shares_pre[party_indices])
            party_change <- sum((price_post[party_indices] / price_pre[party_indices] - 1) * shares_pre[party_indices]) / party_share_pre * 100

            # Consumer Harm (Compensating Variation)
            harm <- try(CV(res), silent = TRUE)
            if (inherits(harm, "try-error")) harm <- NA

            return(data.frame(
                market = e_name,
                draw = d_idx,
                industry_price_change = ind_change,
                party_price_change = party_change,
                consumer_harm = harm
            ))
        })

        all_results[[e_idx]] <- bind_rows(market_draw_results)

        if (e_idx %% 50 == 0) cat(sprintf("    Processed %d/%d markets...\n", e_idx, n_events))
    }

    # Finalize model results
    model_results <- bind_rows(all_results)
    model_results$supply_model <- m_name

    saveRDS(model_results, file = file.path(resdir, paste0("effects_distribution_", m_name, ".rds")))
    cat("  Saved results to results/effects_distribution_", m_name, ".rds\n")

    # Cleanup memory
    rm(fit, draws, model_results, all_results)
    gc()
}

cat("\nAll post-merger effect distributions computed!\n")
