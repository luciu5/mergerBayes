functions {
  // ---------------------------------------------------------------------------
  // 1. SOFT HMT PENALTY (One-Sided)
  // ---------------------------------------------------------------------------
  real soft_hmt_penalty(real agg_elast, real max_elast, real scale) {
    if (agg_elast > max_elast) {
      return normal_lpdf(agg_elast | max_elast, scale) - normal_lpdf(max_elast | max_elast, scale);
    } 
    return 0.0; 
  }

  // ---------------------------------------------------------------------------
  // 2. OPTIMIZED JOINT LIKELIHOOD FUNCTION (DECOUPLED + CUTOFF)
  // ---------------------------------------------------------------------------
  real partial_sum_fast(array[] int slice_idx,
                        int start, int end,
                        // --- DATA ---
                        vector logshareIn, vector marginInv, 
                        vector shareIn, vector rateDiff,
                        array[] int event, array[] int tophold, array[] int year,
                        // --- PARAMETERS ---
                        vector a_event, vector b_event, 
                        vector b_tophold_raw, 
                        real sigma_b_strat,
                        real mu_b_fringe, real sigma_b_fringe,
                        
                        vector s0, vector year_effect_demand, vector year_effect_supply,
                        vector cutoff_share, 
                        // --- SCALARS ---
                        real sigma_share_abs, real sigma_margin, 
                        real rateDiff_sd,
                        int supply_model, int use_cutoff) {
    
    real lp = 0;
    int N_slice = end - start + 1;
    // SMOOTHING
    real k = 0.2; 
    
    for (n in 1:N_slice) {
      int i = slice_idx[n];
      int e = event[i];

      // --- CUTOFF LOGIC ---
      real w = 0; // Default: Strategic
      if (use_cutoff == 1) {
        real x = -(shareIn[i] - cutoff_share[e]) / k;
        if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
      }
      
      // Dynamic Mean Valuation: Mix Fringe/Strategic
      real mu_eff    = w * mu_b_fringe; // Strategic base (mu_b_strat) is 0.0
      real sigma_eff = w * sigma_b_fringe + (1-w) * sigma_b_strat;
      real b_quality = mu_eff + sigma_eff * b_tophold_raw[tophold[i]];

      // --- A. DEMAND MEAN ---
      // mu1 is log(share)
      real mu1 = s0[e] + 
                 b_event[e] + 
                 b_quality + 
                 year_effect_demand[year[i]] + 
                 (a_event[e] * rateDiff[i]);

      // --- B. SUPPLY MEAN ---
      real s = shareIn[i];
      real s_out = inv_logit(s0[e]);
      if (s_out < 0.01) s_out = 0.01;
      
      real inv_markup;
      real alpha_structural = a_event[e] / rateDiff_sd;
      
      if (supply_model == 1) { 
        // Bertrand (Hybrid)
        inv_markup = alpha_structural * (w + (1-w) * (1.0 - s * (1.0 - s_out)));
      } else if (supply_model == 2) { 
        // Auction
        real denom = s * (1.0 - s_out);
        if (denom > 0.99) denom = 0.99;
        real log_term = -log(1.0 - denom);
        if (s < 0.001) inv_markup = alpha_structural * (w + (1-w)); 
        else inv_markup = alpha_structural * (w + (1-w) * (denom / log_term)); 
      } else if (supply_model == 3) {
        // Cournot
        inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + s) ));
      } else {
        // MonComp
        inv_markup = alpha_structural;
      }
      
      real mu2 = inv_markup + year_effect_supply[year[i]];

      // --- C. ACCUMULATE ---
      // 1. DEMAND: Absolute Error on Shares
      lp += normal_lpdf(shareIn[i] | exp(mu1), sigma_share_abs); 

      // 2. SUPPLY: Independent Margin Error
      lp += normal_lpdf(marginInv[i] | mu2, sigma_margin);
    }
    return lp;
  }
}

data {
  int<lower=1> N;
  vector[N] shareIn;
  vector<lower=0>[N] marginInv; 
  vector[N] rateDiff;
  real<lower=0> rateDiff_sd;
  
  int<lower=1> N_event;
  array[N] int<lower=1> event;
  int<lower=1> N_tophold;
  array[N] int<lower=1> tophold;
  int<lower=1> N_year;
  array[N] int<lower=1> year;

  vector[N_event] log_deposits;

  int<lower=1, upper=4> supply_model; 
  int<lower=0, upper=1> use_cutoff;
  int<lower=1> grainsize;
  
  int<lower=0, upper=1> is_single_market;  
  int<lower=0, upper=1> use_hmt;           
  int<lower=0, upper=1> fix_supply_intercept; 
  
  real avg_price_hmt;   
  real avg_margin_hmt;
  real ssnip_hmt;   

  real<lower=0> prior_sigma_share;
  real<lower=0> prior_sigma_margin;
  real<lower=0> prior_sigma_meanval_strat; 
  real<lower=0> prior_sigma_meanval_fringe; 
  
  // Model-Specific Alpha Priors
  real prior_alpha_mean;
  real<lower=0> prior_alpha_sd;

  
  vector<lower=0, upper=1>[N_event] min_s0; 
}

transformed data {
  vector[N] logshareIn = log(shareIn);
  array[N] int seq;
  for (i in 1:N) seq[i] = i;
  int N_cutoff = (use_cutoff == 1) ? N_event : 0;
  int N_supply_int = (fix_supply_intercept == 1) ? 0 : 1; 
}

parameters {
  real mu_year_demand;
  real<lower=0> sigma_year_demand;
  vector[N_year] year_raw_demand;
  
  real mu_b_event;
  real<lower=0> sigma_b_event;
  vector[N_event] b_event_raw;
  
  vector[N_tophold] b_tophold_raw;
  real<lower=0> sigma_b_strat;
  real mu_b_fringe;
  real<lower=0> sigma_b_fringe;
  
  real mu_log_a;
  real<lower=0> sigma_log_a;
  vector[N_event] r_event_a_raw;

  vector[N_supply_int] mu_year_supply_raw;
  real<lower=0> sigma_year_supply;
  vector[N_year] year_raw_supply;
  
  vector<lower=0>[N_event] s0_offset; 
  vector<lower=0, upper=1>[N_cutoff] cutoff_share;

  // DECOUPLED PARAMETERS
  real<lower=0> sigma_share_abs; // Absolute error
  real<lower=0> sigma_margin;    // Margin error
}

transformed parameters {
  vector[N_year] year_effect_demand = mu_year_demand + sigma_year_demand * year_raw_demand;
  real mu_year_supply_scalar = (fix_supply_intercept == 1) ? 0.0 : mu_year_supply_raw[1];
  vector[N_year] year_effect_supply = mu_year_supply_scalar + sigma_year_supply * year_raw_supply;
  vector[N_event] a_event = exp(mu_log_a + sigma_log_a * r_event_a_raw);
  
  vector[N_event] b_event;
  if (is_single_market == 1) {
     b_event = rep_vector(0.0, N_event); 
  } else {
     b_event = mu_b_event + sigma_b_event * b_event_raw;
  }
  
  vector[N_event] s0;
  for (e in 1:N_event) {
     s0[e] = logit(fmax(min_s0[e], 1e-6)) + s0_offset[e];
  }
}

model {
  // --- Priors ---
  if (is_single_market == 1) {
    sigma_log_a ~ normal(0, 0.001); 
    // Use passed-in prior for the single alpha
    mu_log_a ~ normal(log(prior_alpha_mean), prior_alpha_sd); 
    sigma_b_event ~ normal(0, 0.001);
  } else {
    sigma_log_a ~ exponential(1); 
    // Use passed-in prior for the global mean alpha
    mu_log_a ~ normal(log(prior_alpha_mean), prior_alpha_sd); 
    sigma_b_event ~ normal(0, 0.5); 
  }
  r_event_a_raw ~ std_normal();
  b_event_raw ~ std_normal();
  mu_b_event ~ normal(0, 1.0); // FIXED: Added prior to prevent drift
  b_tophold_raw ~ std_normal();
  
  sigma_b_strat ~ normal(0, prior_sigma_meanval_strat); 
  sigma_b_fringe ~ normal(0, prior_sigma_meanval_fringe); 
  mu_b_fringe ~ normal(0, 1);
  
  if (N_year == 1) {
      sigma_year_demand ~ normal(0, 0.001); year_raw_demand ~ normal(0, 0.001);
      sigma_year_supply ~ normal(0, 0.001); year_raw_supply ~ normal(0, 0.001);
  } else {
      sigma_year_demand ~ exponential(1); year_raw_demand ~ std_normal();
      sigma_year_supply ~ exponential(1); year_raw_supply ~ std_normal();
  }
  mu_year_demand ~ std_normal();
  
  if (fix_supply_intercept == 0) {
    mu_year_supply_raw ~ std_normal();
  } 

  s0_offset ~ normal(0, 2.0);

  if (use_hmt == 1) {
     real alpha_check = exp(mu_log_a) / rateDiff_sd; 
     real s0_check = mean(inv_logit(s0));
     real agg_elasticity = abs(alpha_check * avg_price_hmt * s0_check);
     real max_elasticity = 1.0 / (ssnip_hmt + avg_margin_hmt);
     target += soft_hmt_penalty(agg_elasticity, max_elasticity, 0.1);
  }
  
  if (use_cutoff == 1) cutoff_share ~ beta(3, 100);
  
  // DECOUPLED PRIORS
  sigma_share_abs ~ normal(0, prior_sigma_share); 
  sigma_margin ~ normal(0, prior_sigma_margin); 

  target += reduce_sum(
    partial_sum_fast,
    seq, grainsize,
    logshareIn, marginInv, shareIn, rateDiff,
    event, tophold, year,
    a_event, b_event, 
    b_tophold_raw, sigma_b_strat, mu_b_fringe, sigma_b_fringe,
    s0, year_effect_demand, year_effect_supply, cutoff_share, 
    sigma_share_abs, sigma_margin,  
    rateDiff_sd,
    supply_model, use_cutoff
  );
}

generated quantities {
  vector[N] log_lik;
  vector[N] pred_logshareIn;
  vector[N] pred_marginInv;
  
  {
     real k = 0.2; 

     for (n in 1:N) {
       int e = event[n];
       
       real w = 0; 
       if (use_cutoff == 1) {
         real x = -(shareIn[n] - cutoff_share[e]) / k;
         if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
       }

       real mu_eff    = w * mu_b_fringe; // Strategic base is 0.0
       real sigma_eff = w * sigma_b_fringe + (1-w) * sigma_b_strat;
       real b_quality = mu_eff + sigma_eff * b_tophold_raw[tophold[n]];
       
       real mu1 = s0[e] + b_event[e] + b_quality + 
                  year_effect_demand[year[n]] + (a_event[e] * rateDiff[n]);
       
       real alpha_structural = a_event[e] / rateDiff_sd;
       real inv_markup;
       real s = shareIn[n];
       real s_out = inv_logit(s0[e]);
       if (s_out < 0.01) s_out = 0.01;
       
       if (supply_model == 1) { 
         inv_markup = alpha_structural * (w + (1-w) * (1.0 - s * (1.0 - s_out)));
       } else if (supply_model == 2) { 
         inv_markup = alpha_structural * (w + (1-w)); 
       } else if (supply_model == 3) {
         inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + s) ));
       } else {
         inv_markup = alpha_structural;
       }
       
       real mu2 = inv_markup + year_effect_supply[year[n]];
       
       pred_logshareIn[n] = mu1;
       pred_marginInv[n] = mu2;
       
       log_lik[n] = normal_lpdf(shareIn[n] | exp(mu1), sigma_share_abs) + 
                    normal_lpdf(marginInv[n] | mu2, sigma_margin);
     }
  }
}
