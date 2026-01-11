### 1. Load Packages
using CSV, DataFrames
using Turing
using Distributions
using StatsPlots
using LinearAlgebra
using Random
using JSON

# Optional: speed up Turing
#Turing.setadbackend(:forwarddiff)
#Turing.setrdcache(true)

### 2. Load and Prepare stan_data
# NOTE: replace this with your actual data file
# The CSV should include columns: shareIn, marginInv, rateDiff, event, tophold, log_deposits




stan_data = JSON.parsefile("~/supreme/data/stan_data.json")

#data = CSV.read("your_data.csv", DataFrame)
N = nrow(data)

# Convert categorical variables to indices if needed
stan_data.event = categorical(data.event)
stan_data.tophold = categorical(data.tophold)

# Extract required arrays
shareIn = stan_data.shareIn
marginInv = stan_data.marginInv
rateDiff = stan_data.rateDiff

# Create grouping index vectors
event = convert(Vector{Int}, levels(stan_data.event)[stan_data.event])
tophold = convert(Vector{Int}, levels(stan_data.tophold)[stan_data.tophold])
log_deposits = stan_data.log_deposits

N_event = maximum(event)
N_tophold = maximum(tophold)

### 3. Define the Model
@model function demand_margin_model(
    shareIn, marginInv, rateDiff,
    event, tophold, log_deposits;
    N_event, N_tophold)

  # Priors for price coefficient and event effects
  a ~ truncated(Normal(1, 0.7), 0, Inf)
  mu_b_event ~ Normal(0, 2)
  sigma_b_event ~ truncated(Normal(0, 1), 0, Inf)
  b_event_raw ~ filldist(Normal(0,1), N_event)

  mu_b_tophold ~ Normal(0, 2)
  sigma_b_tophold ~ truncated(Normal(0,1), 0, Inf)
  b_tophold_raw ~ filldist(Normal(0,1), N_tophold)

  # Outside share global prior
  mu_s0 ~ Beta(alpha, beta)
  tau_s0 ~ truncated(Normal(0, 0.5), 0, Inf)
  beta_deposits ~ Normal(0, 0.5)
  s0_raw ~ filldist(Normal(0,1), N_event)

  # Event-specific sensitivity to rate difference
  sd_event ~ truncated(Normal(0, 0.5), 0, Inf)
  r_event_a_raw ~ filldist(Normal(0,0.5), N_event)

  # Residual correlation structure
  Lrescor ~ LKJCholesky(2, 1.0)

  sigma_logshare ~ truncated(Normal(0,1.4), 0, Inf)
  sigma_margin ~ truncated(Normal(0,0.35), 0, Inf)

  # Transformed (event-level) parameters
  b_event = mu_b_event .+ sigma_b_event .* b_event_raw
  b_tophold = mu_b_tophold .+ sigma_b_tophold .* b_tophold_raw
  a_event = a .+ sd_event .* r_event_a_raw

  # Outside share per event
  s0 = similar(b_event)
  b0 = similar(b_event)
  for i in 1:N_event
    logit_s0 = logit(mu_s0) + beta_deposits * log_deposits[i] + tau_s0 * s0_raw[i]
    s0[i] = logistic(logit_s0)
    b0[i] = logit(s0[i])
  end

  # Likelihood
  for n in 1:length(shareIn)
    event_idx = event[n]
    mu_log = b0[event_idx] + b_event[event_idx] +
             b_tophold[tophold[n]] +
             a_event[event_idx] * rateDiff[n]

    mu_marg = a_event[event_idx] * (1 - shareIn[n] * (1 - s0[event_idx]))
    ? = diagm([sigma_logshare, sigma_margin]) * Lrescor *
        transpose(diagm([sigma_logshare, sigma_margin]) * Lrescor)
    [log(shareIn[n]), marginInv[n]] ~ MvNormal([mu_log, mu_marg], ?)
  end
end


### 4. Run the Model
model = logit_margin_model(N, shareIn, marginInv, rateDiff, event, tophold, log_deposits,
                           N_event, N_tophold, 4.0, 10.0)

# For reproducibility
Random.seed!(1234)

# Run with 4 chains, 1000 warmup, 1000 sampling
chain = sample(model, NUTS(), MCMCThreads(), 4, 2000; discard_adapt=true)

### 5. Inspect Output
using MCMCChains

println(chain)
# plot(chain)  # optional visual inspection
