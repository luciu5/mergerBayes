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
  // 2. OPTIMIZED JOINT LIKELIHOOD FUNCTION
  // ---------------------------------------------------------------------------
  real partial_sum_fast(array[] int slice_idx,
                        int start, int end,
                        // --- DATA ---
                        vector logshareIn, vector marginInv, 
                        vector shareIn, vector rateDiff, 
                        // REMOVED LOAN_RATE
                        array[] int event, array[] int tophold, array[] int year,
                        // --- PARAMETERS ---
                        vector a_event, vector b_event, vector b_tophold_scaled,
                        vector s0, vector year_effect_demand, vector year_effect_supply,
                        vector cutoff_share, 
                        // REMOVED GAMMA_LOAN
                        // --- SCALARS ---
                        real sigma_logshare, real sigma_margin, real rho,
                        real rateDiff_sd,
                        int supply_model, int use_cutoff) {
    
    real lp = 0;
    int N_slice = end - start + 1;
    
    // Pre-calculate Conditional Probability Constants
    real sigma_cond = sigma_margin * sqrt(1.0 - square(rho));
    real slope_cond = rho * (sigma_margin / sigma_logshare);
    real k = 0.05; 
    
    for (n in 1:N_slice) {
      int i = slice_idx[n];
      int e = event[i];

      // --- A. DEMAND MEAN ---
      real mu1 = s0[e] + 
                 b_event[e] + 
                 b_tophold_scaled[tophold[i]] + 
                 year_effect_demand[year[i]] + 
                 (a_event[e] * rateDiff[i]);

      // --- B. SUPPLY MEAN ---
      real w = 0;
      if (use_cutoff == 1) {
        real x = -(shareIn[i] - cutoff_share[e]) / k;
        if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
      }
      
      real s = shareIn[i];
      real s_out = inv_logit(s0[e]);
      if (s_out < 0.01) s_out = 0.01;
      
      real inv_markup;
      real alpha_structural = a_event[e] / rateDiff_sd;
      
      if (supply_model == 1) { 
        // Bertrand
        inv_markup = alpha_structural * (w + (1-w) * (1.0 - s * (1.0 - s_out)));
      } else if (supply_model == 2) { 
        // Auction
        real denom = s * (1.0 - s_out);
        if (denom > 0.99) denom = 0.99;
        real log_term = -log(1.0 - denom);
        if (s < 0.001) inv_markup = alpha_structural * (w + (1-w)); 
        else inv_markup = alpha_structural * (w - (1-w) * (denom / log_term));
      } else if (supply_model == 3) {
        // Cournot
        inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + s) ));
      } else {
        // MonComp
        inv_markup = alpha_structural;
      }
      
      // Cost Shifter: REMOVED LOAN RATE. Only Intercept.
      real mu2 = inv_markup + year_effect_supply[year[i]];

      // --- C. ACCUMULATE ---
      real y1 = logshareIn[i];
      real y2 = marginInv[i]; 
      
      real mu2_cond = mu2 + slope_cond * (y1 - mu1);
      
      lp += normal_lpdf(y1 | mu1, sigma_logshare);
      lp += normal_lpdf(y2 | mu2_cond, sigma_cond);
    }
    return lp;
  }
}

data {
  int<lower=1> N;
  
  // Data Vectors
  vector[N] shareIn;
  vector<lower=0>[N] marginInv; 
  vector[N] rateDiff;
  real<lower=0> rateDiff_sd;
  // REMOVED loan_rate
  
  // Indices
  int<lower=1> N_event;
  array[N] int<lower=1> event;
  int<lower=1> N_tophold;
  array[N] int<lower=1> tophold;
  int<lower=1> N_year;
  array[N] int<lower=1> year;

  // Covariates
  vector[N_event] log_deposits;
  // REMOVED log_assets

  // Model Flags
  int<lower=1, upper=4> supply_model; 
  int<lower=0, upper=1> use_cutoff;
  int<lower=1> grainsize;
  
  // --- CONTROL FLAGS ---
  int<lower=0, upper=1> is_single_market;  
  int<lower=0, upper=1> use_hmt;           
  int<lower=0, upper=1> fix_supply_intercept; 
  
  // Data needed for HMT 
  real avg_price_hmt;   
  real avg_margin_hmt;
  real ssnip_hmt;   

  // Prior Scales 
  real<lower=0> prior_sigma_share;
  real<lower=0> prior_sigma_margin;
  real<lower=0> prior_sigma_meanval; 
}

transformed data {
  vector[N] logshareIn = log(shareIn);
  array[N] int seq;
  for (i in 1:N) seq[i] = i;
}

parameters {
  // --- Demand ---
  real mu_year_demand;
  real<lower=0> sigma_year_demand;
  vector[N_year] year_raw_demand;
  
  real mu_b_event;
  real<lower=0> sigma_b_event;
  vector[N_event] b_event_raw;
  
  // Bank FE (Quality)
  real mu_b_tophold;
  real<lower=0> sigma_b_tophold;
  vector[N_tophold] b_tophold_raw;
  
  // REMOVED beta_assets
  
  // Price Sensitivity
  real mu_log_a;
  real<lower=0> sigma_log_a;
  vector[N_event] r_event_a_raw;

  // --- Supply ---
  real mu_year_supply;
  real<lower=0> sigma_year_supply;
  vector[N_year] year_raw_supply;
  
  // REMOVED gamma_loan
  
  // Latent s0
  real<lower=-6.9, upper=4.6> logit_mu_s0;
  real<lower=0> tau_s0;
  real beta_deposits;
  vector[N_event] s0_raw;
  
  vector<lower=0, upper=1>[N_event] cutoff_share;

  cholesky_factor_corr[2] Lrescor; 
  real<lower=0> sigma_logshare;    
  real<lower=0> sigma_margin;      
}

transformed parameters {
  
  corr_matrix[2] Corr;
  real rho;                
  Corr = multiply_lower_tri_self_transpose(Lrescor);
  rho = Corr[1,2];

  vector[N_year] year_effect_demand = mu_year_demand + sigma_year_demand * year_raw_demand;
  vector[N_year] year_effect_supply = mu_year_supply + sigma_year_supply * year_raw_supply;
  
  vector[N_event] a_event = exp(mu_log_a + sigma_log_a * r_event_a_raw);
  
  vector[N_event] b_event;
  if (is_single_market == 1) {
     b_event = rep_vector(0.0, N_event); 
  } else {
     b_event = mu_b_event + sigma_b_event * b_event_raw;
  }
  
  // REMOVED beta_assets
  vector[N_tophold] b_tophold_scaled = mu_b_tophold + sigma_b_tophold * b_tophold_raw;
  
  vector[N_event] s0 = logit_mu_s0 + beta_deposits * log_deposits + tau_s0 * s0_raw;
}

model {
  // --- Priors ---
  if (is_single_market == 1) {
    sigma_log_a ~ normal(0, 0.001); 
    mu_log_a ~ normal(0, 0.5); 
    sigma_b_event ~ normal(0, 0.001);
  } else {
    sigma_log_a ~ exponential(1); 
    mu_log_a ~ normal(0, 0.5); 
    sigma_b_event ~ normal(0, 0.5); 
  }
  r_event_a_raw ~ std_normal();
  b_event_raw ~ std_normal();
  
  // Bank Heterogeneity: Scaled by prior_sigma_meanval
  sigma_b_tophold ~ normal(0, prior_sigma_meanval); 
  b_tophold_raw ~ std_normal();
  mu_b_tophold ~ normal(0, 1);
  
  // REMOVED beta_assets prior
  
  // Time
  if (N_year == 1) {
      sigma_year_demand ~ normal(0, 0.001); year_raw_demand ~ normal(0, 0.001);
      sigma_year_supply ~ normal(0, 0.001); year_raw_supply ~ normal(0, 0.001);
  } else {
      sigma_year_demand ~ exponential(1); year_raw_demand ~ std_normal();
      sigma_year_supply ~ exponential(1); year_raw_supply ~ std_normal();
  }
  mu_year_demand ~ std_normal();
  
  if (fix_supply_intercept == 1) {
    mu_year_supply ~ normal(0, 0.001); 
  } else {
    mu_year_supply ~ std_normal();
  }

  logit_mu_s0 ~ normal(-0.7, 1.0); 
  tau_s0 ~ normal(0, 0.5); s0_raw ~ std_normal();
  beta_deposits ~ normal(0, 0.5); 

  // HMT
  if (use_hmt == 1) {
     real alpha_check = exp(mu_log_a) / rateDiff_sd; 
     real s0_check = inv_logit(logit_mu_s0);
     real agg_elasticity = abs(alpha_check * avg_price_hmt * s0_check);
     real max_elasticity = 1.0 / (ssnip_hmt + avg_margin_hmt);
     target += soft_hmt_penalty(agg_elasticity, max_elasticity, 0.01);
  }
  
  if (use_cutoff == 1) cutoff_share ~ beta(3, 100);
  
  // REMOVED gamma_loan prior
  
  Lrescor ~ lkj_corr_cholesky(2.0); 
  sigma_logshare ~ normal(0, prior_sigma_share);
  sigma_margin ~ normal(0, prior_sigma_margin); 

  target += reduce_sum(
    partial_sum_fast,
    seq, grainsize,
    logshareIn, marginInv, shareIn, rateDiff, 
    // REMOVED loan_rate
    event, tophold, year,
    a_event, b_event, b_tophold_scaled, s0, 
    year_effect_demand, year_effect_supply, cutoff_share, 
    // REMOVED gamma_loan
    sigma_logshare, sigma_margin, rho,
    rateDiff_sd,
    supply_model, use_cutoff
  );
}

generated quantities {
  corr_matrix[2] Rescor = multiply_lower_tri_self_transpose(Lrescor);
  real rho_gen = Rescor[1, 2];
  
  // LOO Cross-validation inputs
  vector[N] log_lik;
  vector[N] pred_logshareIn;
  vector[N] pred_marginInv;
  
  {
     real sigma_cond = sigma_margin * sqrt(1.0 - square(rho));
     real slope_cond = rho * (sigma_margin / sigma_logshare);

     for (n in 1:N) {
       int e = event[n];
       real mu1 = s0[e] + b_event[e] + b_tophold_scaled[tophold[n]] + 
                  year_effect_demand[year[n]] + (a_event[e] * rateDiff[n]);
       
       real alpha_structural = a_event[e] / rateDiff_sd;
       real inv_markup;
       
       // Re-calculate supply mean for likelihood
       real s = shareIn[n];
       real s_out = inv_logit(s0[e]);
       if (s_out < 0.01) s_out = 0.01;
       
       // --- FIXED SUPPLY LOGIC ---
       // Note: Removed W (cutoff) matching simplified GQ from user prompt
       if (supply_model == 1) { 
         inv_markup = alpha_structural * (1.0 - s * (1.0 - s_out));
       } else if (supply_model == 2) { 
         // Auction
         real denom = s * (1.0 - s_out);
         if (denom > 0.99) denom = 0.99;
         real log_term = -log(1.0 - denom);
         if (s < 0.001) inv_markup = alpha_structural; 
         else inv_markup = alpha_structural * (denom / log_term); 
       } else if (supply_model == 3) {
         inv_markup = alpha_structural * (s_out / (s_out + s));
       } else {
         inv_markup = alpha_structural;
       }
       
       // REMOVED gamma_loan * loan_rate
       real mu2 = inv_markup + year_effect_supply[year[n]];
       
       pred_logshareIn[n] = mu1;
       pred_marginInv[n] = mu2;
       
       real mu2_cond = mu2 + slope_cond * (logshareIn[n] - mu1);
       
       log_lik[n] = normal_lpdf(logshareIn[n] | mu1, sigma_logshare) + 
                    normal_lpdf(marginInv[n] | mu2_cond, sigma_cond);
     }
  }
}
