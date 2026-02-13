functions {
  real soft_hmt_penalty(real agg_elast, real max_elast, real scale) {
    if (agg_elast > max_elast) {
      return normal_lpdf(agg_elast | max_elast, scale) - normal_lpdf(max_elast | max_elast, scale);
    } 
    return 0.0; 
  }

  real partial_sum_fast(array[] int slice_idx, int start, int end,
                        vector marginInv, vector shareIn, vector rateDiff,
                        array[] int event, array[] int tophold, array[] int year, array[] int market_year_idx,
                        vector a_event, vector b_event, vector b_quality_all, 
                        real sigma_b_strat, real mu_b_fringe, real sigma_b_fringe,
                        vector s0,
                        vector cutoff_share, real sigma_share, real sigma_margin, real rho,
                        real rateDiff_sd, int supply_model, int use_cutoff, int is_single_market) {
    
    real lp = 0;
    int N_slice = end - start + 1;
    real sigma_cond = sigma_margin * sqrt(1.0 - square(rho));
    real slope_cond = rho * (sigma_margin / sigma_share);
    real k = 0.2; 
    
    for (n in 1:N_slice) {
      int i = slice_idx[n];
      int e = event[i];
      int mky = market_year_idx[i];

      // SAFETY FLOOR: Consistent with bayesian_single
      real s_out = inv_logit(s0[mky]);
      if (s_out < 0.01) s_out = 0.01; 
      
      real w = 0; 
      if (use_cutoff == 1) {
        real x = -(shareIn[i] - cutoff_share[e]) / k;
        if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
      }
      
      real b_quality = b_quality_all[tophold[i]] + (w * mu_b_fringe);
      real delta = b_event[e] + b_quality + (a_event[e] * rateDiff[i]);
      
      // STABLE PREDICTED SHARE (Absolute Volume Space)
      // We use the logit(s0) as the log-sum intercept
      real mu1 = s0[mky] + delta;
      real pred_s = exp(mu1); 

      real alpha_structural = a_event[e] / rateDiff_sd;
      
      real inv_markup;
      if (supply_model == 1) { 
        inv_markup = alpha_structural * (w + (1-w) * (1.0 - shareIn[i] * (1.0 - s_out)));
      } else if (supply_model == 2) { 
        real denom = fmin(0.99, shareIn[i] * (1.0 - s_out));
        inv_markup = alpha_structural * (w + (1-w) * (denom / -log(1.0 - denom)));
      } else if (supply_model == 3) {
        inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + shareIn[i])));
      } else {
        inv_markup = alpha_structural;
      }
      
      real mu2 = inv_markup;
      real err_share = shareIn[i] - pred_s;
      real mu2_cond = mu2 + slope_cond * err_share;
      
      // ABSOLUTE ERROR: No Dirac-Logit Collapse
      lp += normal_lpdf(shareIn[i] | pred_s, sigma_share); 
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
  int<lower=1, upper=4> supply_model; 
  int<lower=0, upper=1> use_cutoff;
  int<lower=1> grainsize;
  int<lower=1> N_market_year;
  array[N] int<lower=1> market_year_idx;
  int<lower=0> K_s0;        
  matrix[N_market_year, K_s0] X_s0; 
  array[N_market_year] int<lower=1> mky_to_event;
  array[N_market_year] int<lower=1> mky_to_year;
  int<lower=0, upper=1> is_single_market;  
  int<lower=0, upper=1> use_rho; 
  int<lower=0, upper=1> use_hmt;
  int<lower=0, upper=1> fix_supply_intercept; 
  real avg_price_hmt; 
  real avg_margin_hmt; 
  real ssnip_hmt;   
  real<lower=0> prior_sigma_share; 
  real<lower=0> prior_sigma_margin;
  real<lower=0> prior_sigma_meanval_strat; 
  real<lower=0> prior_sigma_meanval_fringe; 
  real prior_alpha_mean; 
  real<lower=0> prior_alpha_sd; 
  real s0_prior_mean; 
  real<lower=0> s0_prior_sd; 
  real<lower=0> prior_sigma_alpha; 
  real<lower=0> prior_sigma_beta_s0; 
  real<lower=0> prior_lkj;
  vector<lower=0, upper=1>[N_market_year] min_s0; 
}

transformed data {
  array[N] int seq;
  for (i in 1:N) seq[i] = i;
}

parameters {
  vector[is_single_market == 0 ? 1 : 0] mu_b_event_vec;
  vector<lower=0>[is_single_market == 0 ? 1 : 0] sigma_b_event_vec;
  vector[is_single_market == 0 ? N_event : 0] b_event_raw;
  vector[N_tophold] b_tophold_raw;
  real<lower=0> sigma_b_strat;
  vector[use_cutoff == 1 ? 1 : 0] mu_b_fringe_vec;
  vector<lower=0>[use_cutoff == 1 ? 1 : 0] sigma_b_fringe_vec;
  real mu_log_a;
  vector<lower=0>[is_single_market == 0 ? 1 : 0] sigma_log_a_vec;
  vector[is_single_market == 0 ? N_event : 0] r_event_a_raw;
  vector<lower=0>[N_market_year] s0_offset; 
  vector<lower=0, upper=1>[(use_cutoff == 1) ? N_event : 0] cutoff_share;
  vector<lower=-1, upper=1>[use_rho == 1 ? 1 : 0] rho_param; 
  real<lower=0> sigma_share;    
  real<lower=0> sigma_margin;     
  vector<lower=0>[N_market_year > 1 ? 1 : 0] sigma_s0_vec; // Residual s0 scale
  
  // Hierarchical S0
  vector[is_single_market == 0 ? N_event : 0] mu_s0_event_raw;
  vector<lower=0>[is_single_market == 0 ? 1 : 0] sigma_s0_event;
  vector[N_year > 1 ? N_year : 0] year_effect_s0_raw;
  vector<lower=0>[N_year > 1 ? 1 : 0] sigma_year_s0;
  
  vector[K_s0] beta_s0;
}

transformed parameters {
  real rho = 0.0;
  if (use_rho == 1) rho = rho_param[1];
  
  real sigma_log_a = 0.0;
  if (is_single_market == 0) sigma_log_a = sigma_log_a_vec[1];
  
  vector[N_event] a_event;
  if (is_single_market == 1) {
    a_event = rep_vector(exp(mu_log_a), N_event);
  } else {
    a_event = exp(mu_log_a + sigma_log_a * r_event_a_raw);
  }
  
  vector[N_event] b_event;
  if (is_single_market == 1) {
    b_event = rep_vector(0.0, N_event);
  } else {
    b_event = mu_b_event_vec[1] + sigma_b_event_vec[1] * b_event_raw;
  }
  vector[N_tophold] b_quality_all = sigma_b_strat * b_tophold_raw;
  
  // Fringe effects: Identified only if cutoff is active
  real mu_b_fringe = 0.0;
  real sigma_b_fringe = 0.0;
  if (use_cutoff == 1) {
    mu_b_fringe = mu_b_fringe_vec[1];
    sigma_b_fringe = sigma_b_fringe_vec[1];
  }

  vector[is_single_market == 0 ? N_event : 0] mu_s0_event;
  if (is_single_market == 1) {
    mu_s0_event = rep_vector(0.0, 0);
  } else {
    mu_s0_event = sigma_s0_event[1] * mu_s0_event_raw;
  }
  
  vector[N_year > 1 ? N_year : 0] year_effect_s0;
  if (N_year > 1) {
    year_effect_s0 = sigma_year_s0[1] * year_effect_s0_raw;
  } else {
    year_effect_s0 = rep_vector(0.0, 0);
  }

  vector[N_market_year] s0;
  for (m in 1:N_market_year) {
     int e = mky_to_event[m];
     int t = mky_to_year[m];
     real base_logit = logit(fmax(min_s0[m], 1e-6));
     real x_beta = (K_s0 > 0) ? (X_s0[m] * beta_s0) : 0.0;
     real mkt_effect = 0.0;
     if (is_single_market == 0) mkt_effect = mu_s0_event[e];
     
     real yr_effect = 0.0;
     if (N_year > 1) yr_effect = year_effect_s0[t];
     
     s0[m] = base_logit + x_beta + mkt_effect + yr_effect + s0_offset[m];
  }
}

model {
  mu_log_a ~ normal(log(prior_alpha_mean), prior_alpha_sd);
  if (is_single_market == 1) {
    // No prior needed for sigma_log_a
  } else {
    sigma_log_a_vec ~ normal(0, 0.5); 
    r_event_a_raw ~ std_normal();
    mu_b_event_vec ~ normal(0, 1.0); 
    b_event_raw ~ std_normal();
    sigma_b_event_vec ~ normal(0, 0.5); 
  }
  b_tophold_raw ~ std_normal();
  
  sigma_b_strat ~ normal(0, prior_sigma_meanval_strat);
  if (use_cutoff == 1) {
    sigma_b_fringe_vec ~ normal(0, prior_sigma_meanval_fringe);
    mu_b_fringe_vec ~ normal(0, 2);
  }
  
  if (N_market_year > 1) {
    s0_offset ~ normal(0, sigma_s0_vec[1]); 
    sigma_s0_vec ~ normal(0, 1.0);
  } else {
    s0_offset ~ normal(0, 5.0); // Non-hierarchical broad prior
  }
  beta_s0 ~ normal(0, prior_sigma_beta_s0);

  if (is_single_market == 0) {
    mu_s0_event_raw ~ std_normal();
    sigma_s0_event ~ normal(0, 1.0);
  }
  if (N_year > 1) {
    year_effect_s0_raw ~ std_normal();
    sigma_year_s0 ~ normal(0, 1.0);
  }  
  if (use_rho == 1) {
    rho_param ~ uniform(-1, 1); // Equivalent to LKJ(1) for 2x2
  }
  sigma_share ~ normal(0, prior_sigma_share); 
  sigma_margin ~ normal(0, prior_sigma_margin); 

  if (use_hmt == 1) {
     real alpha_check = exp(mu_log_a) / rateDiff_sd; 
     real s0_check = mean(inv_logit(s0));
     real agg_elasticity = abs(alpha_check * avg_price_hmt * s0_check);
     real max_elasticity = 1.0 / (ssnip_hmt + avg_margin_hmt);
     target += soft_hmt_penalty(agg_elasticity, max_elasticity, 0.1);
  }

  target += reduce_sum(partial_sum_fast, seq, grainsize,
                       marginInv, shareIn, rateDiff, event, tophold, year, market_year_idx,
                       a_event, b_event, b_quality_all, sigma_b_strat, mu_b_fringe, sigma_b_fringe,
                       s0, cutoff_share, 
                       sigma_share, sigma_margin, rho, rateDiff_sd, supply_model, use_cutoff, is_single_market);
}

generated quantities {
  vector[N] pred_shareIn;
  vector[N] pred_marginInv;
  
  {
    real k = 0.2; 
    for (n in 1:N) {
      int e = event[n];
      int mky = market_year_idx[n];
      
      real w = 0; 
      if (use_cutoff == 1) {
        real x = -(shareIn[n] - cutoff_share[e]) / k;
        if (x > 10) w = 1.0; else if (x < -10) w = 0.0; else w = inv_logit(x);
      }
      
      real b_quality = b_quality_all[tophold[n]] + (w * mu_b_fringe);
      real delta = b_event[e] + b_quality + (a_event[e] * rateDiff[n]);
      
      real mu1 = s0[mky] + delta;
      pred_shareIn[n] = exp(mu1);
      
      real s_out = inv_logit(s0[mky]);
      if (s_out < 0.01) s_out = 0.01; 
      real alpha_structural = a_event[e] / rateDiff_sd;
      
      real inv_markup;
      if (supply_model == 1) { 
        inv_markup = alpha_structural * (w + (1-w) * (1.0 - shareIn[n] * (1.0 - s_out)));
      } else if (supply_model == 2) { 
        real denom = fmin(0.99, shareIn[n] * (1.0 - s_out));
        inv_markup = alpha_structural * (w + (1-w) * (denom / -log(1.0 - denom)));
      } else if (supply_model == 3) {
        inv_markup = alpha_structural * (w + (1-w) * (s_out / (s_out + shareIn[n])));
      } else {
        inv_markup = alpha_structural;
      }
      
      pred_marginInv[n] = inv_markup;
    }
  }
}


