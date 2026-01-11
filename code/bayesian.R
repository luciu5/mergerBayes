# ---- Load Libraries ----
rm(list=ls())

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
options(mc.cores = chain_count)  # Use 1 core per chain since we'll use threads within each chain
rstan_options(auto_write = TRUE)
Sys.setenv(STAN_NUM_THREADS = thread_count)

# Configure Stan to use multiple threads per chain
rstan_options(threads_per_chain = thread_count)

# add arg to toggle cutoff (1 = enable, 0 = disable)
use_cutoff <- as.numeric(args[4])
if (is.na(use_cutoff)) use_cutoff <- 0

model_name <- c("bertrand","2nd","cournot","moncom")
datadir <- "~m1cst00/supreme/data"
modelpath <- "~m1cst00/Projects/supreme/code/bayes/bayesian.stan"
outfile <- file.path(datadir, paste0("stan_supreme_",model_name[thismodel],".RData"))

load(file=file.path(datadir,"event_overlaps.RData"))

popdata <- data.table::fread(file=file.path(datadir,"nhgis0004_csv","nhgis0004_ds91_1960_county.csv")) %>% mutate(
  fips=STATEA*100+COUNTYA/10,pop=B5O001)

simdata <- data.table::fread(file=file.path(datadir,"call_report","call_report_1960_1968.csv")
                             #,show_col_types = FALSE
) %>%
  dplyr::rename(margin=operating_margin) %>%
  select(year,rssdid,bank_name,fips,total_deposits,total_loans,total_assets,total_ts_deposits,total_demand_deposits,
         net_income,margin,rate_deposits,rate_loans) %>%
  bind_rows(call_report <- data.table::fread(file=file.path(datadir,"call_report","call_report_1969_1978.csv")
                                             #,show_col_types = FALSE
  ) %>%
    dplyr::rename(margin=int_income_margin) %>%
    select(year,rssdid,bank_name,fips,total_deposits,total_loans,total_assets,total_ts_deposits,total_demand_deposits,
           net_income,margin,rate_deposits,rate_loans)
  ) %>%
  filter(year %in% c(1960:1978))

simdata <- inner_join(simdata,overlaps)
events <- expand_grid(events,cntr=seq(-1,5,1)
) %>% mutate(year=year+cntr) #%>% select(-cntr)
simdata <- inner_join(simdata,select(events,event_id,year,cntr))
simdata <- inner_join(simdata,select(popdata,fips,pop))

simdata <- filter(simdata,!(event_id == 6 & !rssdid %in% c(319403,46905,49504,480219,246116,140018,991715)))

simdata <- rename(simdata,tophold=rssdid) %>% 
  mutate(event_id=factor(event_id),
         fips=factor(fips),
         year=factor(year),
         #event=interaction(event_id,fips,drop=TRUE),
         tophold_event=interaction(tophold,event_id,drop=TRUE)) %>% 
  mutate(margin=ifelse(margin<=0 ,NA,margin))


simdata <- simdata %>%
  group_by(event_id,year,tophold) %>%
  mutate(total_q_bank_cty = sum(total_deposits,na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(event_id,year) %>%
  mutate(deposit_total_cty = sum(total_deposits,na.rm = TRUE),
         loan_total_cty = sum(total_loans,na.rm = TRUE),
         asset_total_cty = sum(total_assets,na.rm = TRUE)) %>%
  mutate(shareIn = (total_q_bank_cty / deposit_total_cty)) %>%
  group_by(tophold) %>% 
  mutate(tooSmall=max(shareIn,na.rm=TRUE)<.01) %>% ungroup()


simdata <- filter(simdata,
                  !is.na(margin)
                  # & !tooSmall
) %>%
  mutate(event_id=droplevels(event_id),
         tophold_event=droplevels(tophold_event),
         year=droplevels(year)) %>%
  #filter( event %in% levels(event)[1:100] )%>%
  mutate(event_id=droplevels(event_id),
         #tophold=ifelse(tooSmall,paste0(99999,event_id),as.character(tophold)),
         tophold_event=droplevels(tophold_event),
         tophold=factor(tophold),
         year=droplevels(year)) %>%
  arrange(event_id,year,tophold) %>% distinct()


simdata <- simdata %>%
  mutate(
    event_id = droplevels(event_id),
    tophold = droplevels(tophold),
    year = droplevels(year),
    tophold_event = droplevels(tophold_event)
  )

eventdata <- select(simdata,event_id,year,deposit_total_cty,loan_total_cty,asset_total_cty,pop) %>% select(-year) %>%
  group_by(event_id) %>% summarise(across(everything(),\(x) mean(x, na.rm = TRUE))) %>%
  distinct() %>% arrange(event_id)

topholddata <-  select(simdata,tophold,total_deposits,total_loans,total_assets) %>% 
  group_by(tophold) %>% summarise(across(everything(),\(x) mean(x, na.rm = TRUE))) %>% 
  arrange(tophold)


# Add market-year indices for parallelization
simdata <- simdata %>%
  arrange(event_id, year) %>%
  mutate(market_year = interaction(event_id, year, drop=TRUE))

# Create market-year grouping for parallelization
market_years <- levels(simdata$market_year)
N_market_year <- length(market_years)

# Add market_year_idx to data
simdata <- simdata %>%
  mutate(market_year_idx = as.numeric(market_year))

# Check data dimensions
cat("Number of observations after filtering:", nrow(simdata), "\n")
cat("N_event:", nlevels(simdata$event_id), "\n")
cat("N_tophold:", nlevels(simdata$tophold), "\n")
cat("N_year:", nlevels(simdata$year), "\n")

# Stop if too few observations
if (nrow(simdata) < 10) {
  stop("Too few observations after filtering! Check your filter conditions.")
}



# Create stan_data with parallelization fields
stan_data <- list(
  supply_model=thismodel, # specify 1=Bertrand,2=SSA, 3=Cournot,4=MonCom
  use_cutoff = as.integer(use_cutoff), # 1 enable cutoff, 0 disable
  N = nrow(simdata),
  shareIn = simdata$shareIn,
  marginInv = 1/simdata$margin,
  rateDiff = as.numeric(scale(simdata$rate_deposits, center = TRUE, scale = TRUE)),
  loan_rate= as.numeric(scale(simdata$rate_loans, center = TRUE, scale = TRUE)),
  N_event = nlevels(simdata$event_id),
  event = as.integer(simdata$event_id),
  N_tophold = nlevels(simdata$tophold),
  tophold = as.integer(simdata$tophold),
  log_deposits = as.numeric(scale(log(eventdata$deposit_total_cty), center = TRUE, scale = TRUE)),
  log_assets = as.numeric(scale(log(topholddata$total_assets), center = TRUE, scale = TRUE)),
  rateDiff_sd=sd(simdata$rate_deposits),
  N_year=nlevels(simdata$year),
  year= as.numeric(simdata$year),
  # New parallelization variables
  N_market_year = N_market_year,
  market_year_idx = as.integer(simdata$market_year_idx),
  grainsize=max(1, round(nrow(simdata) / (10 * thread_count)))
)

write_json(stan_data, file.path(datadir,"stan_data.json"), pretty = TRUE, auto_unbox = TRUE)

# Compile the model with threading enabled
model <- stan_model(modelpath)

# Sample from the posterior with threading
system.time(fit <- sampling(
  model,
  data = stan_data,
  chains = chain_count,
  init=0,
  iter = 7000,         
  warmup = 1500,       
  control = list(adapt_delta = 0.90, max_treedepth=12)
)
)


# Build parameter lists for plotting and summaries depending on use_cutoff
# Match Stan parameter names
base_pars <- c("a_event", "logit_mu_s0", "s0", "Rescor", 
               "sigma_logshare", "sigma_margin", "a_event", "gamma_loan")

if (stan_data$use_cutoff == 1) {
  plot_pars <- c(base_pars, "cutoff_share")
  sum_pars  <- c(base_pars, "cutoff_share", "year_effect_demand", "year_effect_supply")
} else {
  plot_pars <- base_pars
  sum_pars  <- c(base_pars, "year_effect_demand", "year_effect_supply")
}

# Plot summaries
plot_sum <- plot(fit, pars = plot_pars)
fit_sum  <- summary(fit, pars = sum_pars)$summary

# Bridge sampling
bridge <- bridge_sampler(fit, cores=thread_count)

# Extract predicted margins
# If you previously used mu_margin, calculate it from logshare and marginInv or from generated quantities
# For now, extract marginInv to inspect predictions
margin_samples <- rstan::extract(fit, "inv_markup_pred")[[1]]
neg_count <- sum(margin_samples < 0)
neg_pct <- mean(margin_samples < 0) * 100
cat("Number of negative predicted margins:", neg_count, "\n")
cat("Percentage of negative predictions:", neg_pct, "%\n")

if(neg_count > 0) {
  cat("Mean of negative predictions:", mean(margin_samples[margin_samples < 0]), "\n")
  cat("Min of negative predictions:", min(margin_samples), "\n")
}


save(fit, bridge, plot_sum, fit_sum, file = outfile)
