functions {
  // ---------------------------------------------------------------------------
  // OPTIMIZED JOINT LIKELIHOOD FUNCTION
  // Calculates P(y1, y2) = P(y1) * P(y2 | y1) for maximum speed.
  // ---------------------------------------------------------------------------
  real partial_sum_fast(int[] slice_idx,
                        int start, int end,
                        // --- DATA ---
                        vector logshareIn, vector marginInv, 
                        vector shareIn, vector rateDiff, vector loan_rate,
                        int[] event, int[] tophold, int[] year,
                        // --- PARAMETERS ---
                        vector a_event, vector b_event, vector b_tophold_scaled,
                        vector s0, vector year_effect_demand, vector year_effect_supply,
                        vector cutoff_share, real gamma_loan,
                        // --- SCALARS ---
                        real sigma_logshare, real sigma_margin, real rho,
                        real rateDiff_sd,
                        int supply_model, int use_cutoff) {
    
    real lp = 0;
    int N_slice = end - start + 1;
    
    // 1. Pre-calculate Conditional Probability Constants
    // y2 | y1 ~ Normal(mu2 + slope * (y1 - mu1), sigma_cond)
    real sigma_cond = sigma_margin * sqrt(1.0 - square(rho));
    real slope_cond = rho * (sigma_margin / sigma_logshare);
    real k = 0.05; // Smoothness for cutoff
    
    for (n in 1:N_slice) {
      int i = slice_idx[n];
      int e = event[i];

      // --- 2. DEMAND MEAN (Log Share) ---
      real mu1 = s0[e] + 
                 b_event[e] + 
                 b_tophold_scaled[tophold[i]] + 
                 year_effect_demand[year[i]] + 
                 (a_event[e] * rateDiff[i]);

      // --- 3. SUPPLY MEAN (Inverse Margin) ---
      
      // Cutoff Logic
      real w = 0;
      if (use_cutoff == 1) {
        real x = -(shareIn[i] - cutoff_share[e]) / k;
        if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
      }
      
      real s = shareIn[i];
      // Clamp s_out to avoid numerical explosions
      real s_out = inv_logit(s0[e]);
      if (s_out < 0.01) s_out = 0.01;
      
      real a = a_event[e];
      real inv_markup;
      
      // Conduct Logic (Inverse Form)
      if (supply_model == 1) { 
        // Structural alpha = a / sd_rate (scales standardized coefficient to proportions)
        real alpha_structural = a / rateDiff_sd;
        
        // BERTRAND: 1/m = alpha * (1 - s(1-s0))
        inv_markup = alpha_structural * (w + (1-w) * (1.0 - s * (1.0 - s_out)));
        
      } else if (supply_model == 2) { 
        // AUCTION: 1/m = alpha * (-log(1-denom)/s)
        real denom = s * (1.0 - s_out);
        if (denom > 0.99) denom = 0.99;
        real log_term = -log(1.0 - denom);
        
        real alpha_structural = a / rateDiff_sd;
        
        if (s < 0.001) inv_markup = alpha_structural * (w + (1-w)); 
        else inv_markup = alpha_structural * (w - (1-w) * (denom / log_term));
        
      } else if (supply_model == 3) {
        real alpha_structural = a / rateDiff_sd;
        
        // COURNOT: 1/m = alpha / (1 + s/s0)
        inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + s) ));
        
      } else {
        real alpha_structural = a / rateDiff_sd;
        
        // MONOPOLISTIC COMP: 1/m = alpha
        inv_markup = alpha_structural;
      }
      
      // Cost Shifter Included: (Deposit Opportunity Cost)
      real mu2 = inv_markup + gamma_loan * loan_rate[i] + year_effect_supply[year[i]];

      // --- 4. ACCUMULATE JOINT LIKELIHOOD ---
      real y1 = logshareIn[i];
      real y2 = marginInv[i]; 
      
      // Conditional Mean
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
  vector<lower=0>[N] marginInv; // Observed 1/Margin
  vector[N] rateDiff;
  real<lower=0> rateDiff_sd;
  vector[N] loan_rate;           // Cost Shifter
  
  // Indices
  int<lower=1> N_event;
  array[N] int<lower=1> event;
  int<lower=1> N_tophold;
  array[N] int<lower=1> tophold;
  int<lower=1> N_year;
  array[N] int<lower=1> year;

  // Covariates
  vector[N_event] log_deposits;
  vector[N_tophold] log_assets;

  // Model Flags
  int<lower=1, upper=4> supply_model; 
  int<lower=0, upper=1> use_cutoff;
  int<lower=1> grainsize;
}

transformed data {
  vector[N] logshareIn = log(shareIn);
  
  // Sequence for reduce_sum
  int seq[N];
  for (i in 1:N) seq[i] = i;
}

parameters {
  // --- Demand Hierarchical ---
  real mu_year_demand;
  real<lower=0> sigma_year_demand;
  vector[N_year] year_raw_demand;
  
  real mu_b_event;
  real<lower=0> sigma_b_event;
  vector[N_event] b_event_raw;
  
  real mu_b_tophold;
  real<lower=0> sigma_b_tophold;
  vector[N_tophold] b_tophold_raw;
  
  // Price Sensitivity (Alpha)
  real mu_log_a;
  real<lower=0> sigma_log_a;
  vector[N_event] r_event_a_raw;

  // --- Supply Hierarchical ---
  real mu_year_supply;
  real<lower=0> sigma_year_supply;
  vector[N_year] year_raw_supply;
  
  real gamma_loan; // Lending Rate Coefficient
  
  // Latent s0
  real<lower=-6.9, upper=4.6> logit_mu_s0;
  real<lower=0> tau_s0;
  real beta_deposits;
  real beta_assets;
  vector[N_event] s0_raw;
  
  vector<lower=0, upper=1>[N_event] cutoff_share;

  // Covariance Parameters
  cholesky_factor_corr[2] Lrescor; 
  real<lower=0> sigma_logshare;    
  real<lower=0> sigma_margin;      
}

transformed parameters {
  
  corr_matrix[2] Corr;
    real rho;                
    Corr = multiply_lower_tri_self_transpose(Lrescor);
    rho = Corr[1,2];

  // 1. Pre-calculate global vectors (Fast, size N_event/N_year)
  vector[N_year] year_effect_demand = mu_year_demand + sigma_year_demand * year_raw_demand;
  vector[N_year] year_effect_supply = mu_year_supply + sigma_year_supply * year_raw_supply;
  
  vector[N_event] a_event = exp(mu_log_a + sigma_log_a * r_event_a_raw);
  vector[N_event] b_event = mu_b_event + sigma_b_event * b_event_raw;
  
  // Includes Asset Size effect on Holding Co FE
  vector[N_tophold] b_tophold_scaled = mu_b_tophold + beta_assets * log_assets + sigma_b_tophold * b_tophold_raw;
  
  // s0 passed as Logit to be stable
  vector[N_event] s0 = logit_mu_s0 + beta_deposits * log_deposits + tau_s0 * s0_raw;
}

model {
  // --- RELAXED PRIORS (improved for convergence) ---
  // Year effects - keep these relatively tight
  mu_year_demand ~ std_normal(); sigma_year_demand ~ normal(0, 0.5); year_raw_demand ~ std_normal();
  mu_year_supply ~ std_normal(); sigma_year_supply ~ normal(0, 0.5); year_raw_supply ~ std_normal();
  
  // Alpha (price coefficient) - RELAXED: was normal(-0.5, 0.5)
  mu_log_a ~ normal(0, 1); sigma_log_a ~ normal(0, 0.5); r_event_a_raw ~ std_normal();
  
  // Market and bank fixed effects - RELAXED: was normal(0, 0.2)
  mu_b_event ~ normal(0, 1); sigma_b_event ~ normal(0, 0.5); b_event_raw ~ std_normal();
  mu_b_tophold ~ normal(0, 1); sigma_b_tophold ~ normal(0, 0.5); b_tophold_raw ~ std_normal();
  
  // Outside share - RELAXED tau: was normal(0, 0.2)
  logit_mu_s0 ~ normal(-1, 0.5); tau_s0 ~ normal(0, 0.5); 
  beta_deposits ~ normal(0, 0.5); beta_assets ~ normal(0, 0.5); s0_raw ~ std_normal();
  
  if (use_cutoff == 1) cutoff_share ~ beta(3, 30);
  
  gamma_loan ~ normal(-1, 1);
  
  // Correlation
  Lrescor ~ lkj_corr_cholesky(4.0); 
  // Outcome variances - RELAXED: was normal(0, 2) and normal(0, 10)
  sigma_logshare ~ normal(0, 3);
  sigma_margin ~ normal(0, 15); 


    
  target += reduce_sum(
    partial_sum_fast,
    seq, grainsize,
    logshareIn, marginInv, shareIn, rateDiff, loan_rate,
    event, tophold, year,
    a_event, b_event, b_tophold_scaled, s0, 
    year_effect_demand, year_effect_supply, cutoff_share, gamma_loan,
    sigma_logshare, sigma_margin, rho,
    rateDiff_sd,
    supply_model, use_cutoff
  );
 
}

generated quantities {
  corr_matrix[2] Rescor = multiply_lower_tri_self_transpose(Lrescor);
  real rho_gen = Rescor[1, 2];
  
  // Reverting to vector for LOO as requested
  vector[N] log_lik;
  
  // REMOVED: vector[N] inv_markup_pred; to save memory (~2.5GB reduction)
  int<lower=0> neg_margin_count = 0;
  
  // Replicate logic for LOO and negative margin counting
  {
     real sigma_cond = sigma_margin * sqrt(1.0 - square(rho));
     real slope_cond = rho * (sigma_margin / sigma_logshare);
     real k = 0.05;

     for (n in 1:N) {
       int e = event[n];
       
       real mu1 = s0[e] + b_event[e] + b_tophold_scaled[tophold[n]] + 
                  year_effect_demand[year[n]] + (a_event[e] * rateDiff[n]);
       
       real w = 0;
       if (use_cutoff == 1) {
         real x = -(shareIn[n] - cutoff_share[e]) / k;
         if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
       }
       
       real s = shareIn[n];
       real s_out = inv_logit(s0[e]);
       if (s_out < 0.01) s_out = 0.01;
       
       real inv_markup;
       real alpha_structural = a_event[e] / rateDiff_sd;
       
       if (supply_model == 1) { 
         inv_markup = alpha_structural * (w + (1-w) * (1 - s * (1 - s_out)));
       } else if (supply_model == 2) { 
         real denom = s * (1 - s_out);
         if (denom > 0.99) denom = 0.99;
         real log_term = -log(1.0 - denom);
         if (s < 0.001) inv_markup = alpha_structural * (w + (1-w));
         else inv_markup = alpha_structural * (w - (1-w) * (denom / log_term ));
       } else if (supply_model == 3) {
         inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + s) ));
       } else {
         inv_markup = alpha_structural;
       }
       
       // Count negative margins directly here instead of saving vector
       if (inv_markup < 0) {
         neg_margin_count += 1;
       }
       
       // Re-include loan_rate in Generated Quantities
       real mu2 = inv_markup + gamma_loan * loan_rate[n] + year_effect_supply[year[n]];
       
       real y1 = logshareIn[n];
       real y2 = marginInv[n];
       real mu2_cond = mu2 + slope_cond * (y1 - mu1);
       
       log_lik[n] = normal_lpdf(y1 | mu1, sigma_logshare) + 
                    normal_lpdf(y2 | mu2_cond, sigma_cond);
     }
  }
}
