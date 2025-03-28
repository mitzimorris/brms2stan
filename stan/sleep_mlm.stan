/**
 * Multilevel linear model for sleep study data
 * Formula: Reaction ~ 1 + Days + (1 + Days | Subject)
 *
 * This model includes:
 * - Global intercept and slope for Days
 * - Subject-specific random intercepts and slopes
 * - Correlation between random effects
 *
 * The model uses a non-centered parameterization with Cholesky factorization
 * for numerical stability and efficiency.
 */
data {
  int<lower=0> N;                     // number of observations
  int<lower=0> J;                     // number of subjects
  array[N] int<lower=1, upper=J> subj;  // subject ID for each observation
  vector[N] day;                      // day predictor (0-9)
  vector[N] y;                 // reaction time outcome
}
parameters {
  real b_intercept;  // global intercept
  real b_day; // global day effect
  real<lower=0> sigma;
  
  // Subject-level effects - non-centered parameterization
  matrix[2, J] z; // standardized random effects
  cholesky_factor_corr[2] L_Omega;  // Cholesky factor of correlation matrix
  vector<lower=0>[2] tau; // scale of random effects
}
transformed parameters {
  // random effects matrix scaled, transposed
  matrix[J, 2] r = (diag_pre_multiply(tau, L_Omega) * z)';
}
model {
  y ~ normal(r[subj,1] + b_intercept + (r[subj,2] + b_day) .* day, sigma);

  b_intercept ~ normal(250, 50);  // weakly informative prior for intercept
  b_day ~ normal(10, 10);         // weakly informative prior for day effect
  
  to_vector(z) ~ std_normal(); 
  tau ~ cauchy(0, 25);
  L_Omega ~ lkj_corr_cholesky(2);
  sigma ~ exponential(1);
}
generated quantities {
  // Reconstruct correlation matrix from Cholesky factor
  matrix[2, 2] Omega;
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  
  real sd_intercept = tau[1];
  real sd_day = tau[2];
  real cor_intercept_day = Omega[1, 2];

  // Posterior predictive - new data y-tilde, given y
  vector[N] y_rep;
  vector[N] log_lik;
  {  // don't save to output
    vector[N] eta = r[subj,2] + b_intercept + (r[subj,2] + b_day) .* day;
    y_rep = to_vector(normal_rng(eta, sigma));
    for (n in 1:N) {
      log_lik[n] = normal_lpdf(y[n] | eta[n], sigma);
    }
  }
}
