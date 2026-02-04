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
                        vector marginInv, 
                        vector shareIn, vector rateDiff,
                        array[] int event, array[] int tophold, array[] int year,
                        // --- PARAMETERS ---
                        vector a_event, vector b_event, 
                        // RE STRUCTURE MODIFIED: Raw params + Split Hyperparams
                        vector b_tophold_raw, 
                        real mu_b_strat, real sigma_b_strat,
                        real mu_b_fringe, real sigma_b_fringe,
                        
                        vector s0, vector year_effect_demand, vector year_effect_supply,
                        vector cutoff_share, 
                        // --- SCALARS ---
                        real sigma_share_abs, real sigma_margin, real rho,
                        real rateDiff_sd,
                        int supply_model, int use_cutoff) {
    
    real lp = 0;
    int N_slice = end - start + 1;
    real sigma_cond = sigma_margin * sqrt(1.0 - square(rho));
    real slope_cond = rho * (sigma_margin / sigma_share_abs);
    // SMOOTHING: Increased k from 0.05 to 0.2 to help Bridge Sampling navigate the switch
    real k = 0.2; 
    
    for (n in 1:N_slice) {
      int i = slice_idx[n];
      int e = event[i];

      // --- HYBRID LOGIC ---
      real w = 0; // Default: Strategic (w=0)
      if (use_cutoff == 1) {
        real x = -(shareIn[i] - cutoff_share[e]) / k;
        if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
      }
      
      // Dynamic Mean Valuation: Mix Fringe/Strategic distributions
      real mu_eff    = w * mu_b_fringe    + (1-w) * mu_b_strat;
      real sigma_eff = w * sigma_b_fringe + (1-w) * sigma_b_strat;
      real b_quality = mu_eff + sigma_eff * b_tophold_raw[tophold[i]];

      // --- A. DEMAND MEAN ---
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
        // Bertrand (Hybrid: w=1 -> MonCom, w=0 -> Bertrand)
        inv_markup = alpha_structural * (w + (1-w) * (1.0 - s * (1.0 - s_out)));
      } else if (supply_model == 2) { 
        // Auction
        real denom = s * (1.0 - s_out);
        if (denom > 0.99) denom = 0.99;
        real log_term = -log(1.0 - denom);
        if (s < 0.001) inv_markup = alpha_structural * (w + (1-w)); 
        else inv_markup = alpha_structural * (w + (1-w) * (denom / log_term)); // Fixed mixing logic
      } else if (supply_model == 3) {
        // Cournot
        inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + s) ));
      } else {
        // MonComp (Always MonCom)
        inv_markup = alpha_structural;
      }
      
      real mu2 = inv_markup + year_effect_supply[year[i]];

      // --- C. ACCUMULATE ---
      // 1. Calculate Expected Share
      real pred_share = exp(mu1);
      
      // 2. Share Error
      real err_share = shareIn[i] - pred_share;
      
      // 3. Conditional Mean for Margin
      real mu2_cond = mu2 + slope_cond * err_share;
      
      // 4. LPDF Accumulation
      lp += normal_lpdf(shareIn[i] | pred_share, sigma_share_abs); 
      lp += normal_lpdf(marginInv[i] | mu2_cond, sigma_cond);
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
  real<lower=0> prior_sigma_meanval_strat; // Renamed
  real<lower=0> prior_sigma_meanval_fringe; // Added
  
  vector<lower=0, upper=1>[N_event] min_s0; // Added: Fringe share lower bound
}

transformed data {
  vector[N] logshareIn = log(shareIn);
  array[N] int seq;
  for (i in 1:N) seq[i] = i;
  // Conditional Parameter Size
  int N_cutoff = (use_cutoff == 1) ? N_event : 0;
  // If fixed, remove intercept param entirely to stabilize Bridge Sampling
  int N_supply_int = (fix_supply_intercept == 1) ? 0 : 1; 
}

parameters {
  real mu_year_demand;
  real<lower=0> sigma_year_demand;
  vector[N_year] year_raw_demand;
  
  real mu_b_event;
  real<lower=0> sigma_b_event;
  vector[N_event] b_event_raw;
  
  // Bank FE (Splitting Mean/Sigma by Type)
  vector[N_tophold] b_tophold_raw;
  real mu_b_strat;
  real<lower=0> sigma_b_strat;
  real mu_b_fringe;
  real<lower=0> sigma_b_fringe;
  
  
  real mu_log_a;
  real<lower=0> sigma_log_a;
  vector[N_event] r_event_a_raw;

  // --- Supply Hierarchical ---
  // Conditional: Only estimate if NOT fixed to 0
  vector[N_supply_int] mu_year_supply_raw;
  
  real<lower=0> sigma_year_supply;
  vector[N_year] year_raw_supply;
  
  // S0: Hard Floor Offset (Positive Constraint)
  vector<lower=0>[N_event] s0_offset; 
  
  vector<lower=0, upper=1>[N_cutoff] cutoff_share;

  cholesky_factor_corr[2] Lrescor; 
  real<lower=0> sigma_share_abs;    
  real<lower=0> sigma_margin;      
}

transformed parameters {
  corr_matrix[2] Corr;
  real rho;                
  Corr = multiply_lower_tri_self_transpose(Lrescor);
  
  // FIX: For single market, force rho to 0 to prevent identification issues
  // The Lrescor parameter remains but is detached from the likelihood
  rho = (is_single_market == 1) ? 0.0 : Corr[1,2];

  vector[N_year] year_effect_demand = mu_year_demand + sigma_year_demand * year_raw_demand;
  
  // Handle Fixed Intercept Logic (Hard Zero vs Estimated)
  real mu_year_supply_scalar = (fix_supply_intercept == 1) ? 0.0 : mu_year_supply_raw[1];
  vector[N_year] year_effect_supply = mu_year_supply_scalar + sigma_year_supply * year_raw_supply;
  
  vector[N_event] a_event = exp(mu_log_a + sigma_log_a * r_event_a_raw);
  
  vector[N_event] b_event;
  if (is_single_market == 1) {
     b_event = rep_vector(0.0, N_event); 
  } else {
     b_event = mu_b_event + sigma_b_event * b_event_raw;
  }
  
  // HARD FLOOR CONSTRAINT:
  // s0 is the logit of the outside share.
  // We enforce s0 > logit(fringe_share) by adding a positive offset.
  vector[N_event] s0;
  for (e in 1:N_event) {
     s0[e] = logit(fmax(min_s0[e], 0.001)) + s0_offset[e];
  }
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
  
  // Bank Heterogeneity: Split Priors
  b_tophold_raw ~ std_normal();
  
  sigma_b_strat ~ normal(0, prior_sigma_meanval_strat); 
  mu_b_strat ~ normal(0, 1);
  
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
  
  // Supply Intercept Prior (Only if estimated)
  if (fix_supply_intercept == 0) {
    mu_year_supply_raw ~ std_normal();
  } 

  // S0 Offset Prior (Half-Normal)
  // Relaxed to allow outside share to be significantly larger than fringe share
  // (e.g. if fringe=5% (logit -3) but outside=30% (logit -1), need offset +2)
  s0_offset ~ normal(0, 2.0); 

  // REMOVED GHOSTS (logit_mu_s0, tau_s0, beta_deposits)
 
  
  if (use_hmt == 1) {
     real alpha_check = exp(mu_log_a) / rateDiff_sd; 
     real s0_check = mean(inv_logit(s0));
     real agg_elasticity = abs(alpha_check * avg_price_hmt * s0_check);
     real max_elasticity = 1.0 / (ssnip_hmt + avg_margin_hmt);
     target += soft_hmt_penalty(agg_elasticity, max_elasticity, 0.1);
  }
  
  if (use_cutoff == 1) cutoff_share ~ beta(3, 100);
  
  Lrescor ~ lkj_corr_cholesky(2.0); 
  sigma_share_abs ~ normal(0, prior_sigma_share);
  sigma_margin ~ normal(0, prior_sigma_margin); 

  target += reduce_sum(
    partial_sum_fast,
    seq, grainsize,
    marginInv, shareIn, rateDiff,
    event, tophold, year,
    a_event, b_event, 
    b_tophold_raw, mu_b_strat, sigma_b_strat, mu_b_fringe, sigma_b_fringe,
    s0, year_effect_demand, year_effect_supply, cutoff_share, 
    sigma_share_abs, sigma_margin, rho,
    rateDiff_sd,
    supply_model, use_cutoff
  );
}

generated quantities {
  corr_matrix[2] Rescor = multiply_lower_tri_self_transpose(Lrescor);
  real rho_gen = rho; // Use the effective rho (0 if single market)
  
  vector[N] log_lik;
  vector[N] pred_shareIn;
  vector[N] pred_marginInv;
  
  {
     real sigma_cond = sigma_margin * sqrt(1.0 - square(rho));
     real slope_cond = rho * (sigma_margin / sigma_share_abs);
     // SMOOTHING: Increased k from 0.05 to 0.2 to help Bridge Sampling navigate the switch
     real k = 0.2; 

     for (n in 1:N) {
       int e = event[n];
       
       // Hybrid W Re-calculation
       real w = 0; 
       if (use_cutoff == 1) {
         real x = -(shareIn[n] - cutoff_share[e]) / k; // Assuming N_event indexing map is valid (cutoff_share size checked)
         // Note: If use_cutoff=1, N_cutoff=N_event. Safe.
         if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
       }

       real mu_eff    = w * mu_b_fringe    + (1-w) * mu_b_strat;
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
         // Auction
         real denom = s * (1.0 - s_out);
         if (denom > 0.99) denom = 0.99;
         real log_term = -log(1.0 - denom);
         if (s < 0.001) inv_markup = alpha_structural * (w + (1-w)); 
         else inv_markup = alpha_structural * (w + (1-w) * (denom / log_term)); 
       } else if (supply_model == 3) {
         inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + s) ));
       } else {
         inv_markup = alpha_structural;
       }
       
       real mu2 = inv_markup + year_effect_supply[year[n]];
       
       // Calculate Expected Share
       real pred_share = exp(mu1);
       pred_shareIn[n] = pred_share;
       pred_marginInv[n] = mu2;
       
       real err_share = shareIn[n] - pred_share;
       real mu2_cond = mu2 + slope_cond * err_share;
       
       log_lik[n] = normal_lpdf(shareIn[n] | pred_share, sigma_share_abs) + 
                    normal_lpdf(marginInv[n] | mu2_cond, sigma_cond);
     }
  }
}
