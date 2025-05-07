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
transformed data {
  int K = 2;  // number of covariates
  real day_mean = mean(day);
  vector[N] day_c = day - day_mean;
    
  matrix[N, 2] x;
  x[ , 1] = rep_vector(1, N);
  x[ , 2] = day_c;
}
parameters {
  real alpha;  // global intercept
  real b_day; // global day effect
  real<lower=0> sigma;
  
  // Subject-level effects - non-centered parameterization
  vector[K] nu;                     // location of beta[ , j]
  vector<lower=0>[K] tau;              // scale of beta[ , j]
  cholesky_factor_corr[K] L_Omega;  // Cholesky factor of correlation matrix
  matrix[K, J] beta_std; // standardized random effects
}
transformed parameters {
  // random effects matrix scaled, transposed  (centered at 0)
  matrix[J, K] beta = (rep_matrix(nu, J) + (diag_pre_multiply(tau, L_Omega) * beta_std))';
}
model {
  vector[N] eta = alpha + day_c * b_day + rows_dot_product(x, beta[ subj, ]);
  y ~ normal(eta, sigma);

  alpha ~ normal(250, 50);
  b_day ~ normal(10, 10);
  sigma ~ exponential(1);
  nu ~ std_normal();
  tau ~ exponential(1);
  L_Omega ~ lkj_corr_cholesky(2);
  to_vector(beta_std) ~ std_normal(); 
}
generated quantities {
  // Reconstruct correlation matrix from Cholesky factor
  matrix[K, K] Sigma = multiply_lower_tri_self_transpose(L_Omega);
  
  real b_intercept = alpha - b_day * day_mean;
  real sd_intercept = tau[1];
  real sd_day = tau[2];
  real cor_intercept_day = Sigma[1, 2];

  // Posterior predictive - new data y-tilde, given y
  vector[N] y_rep;
  vector[N] log_lik;
  {  // don't save to output
    vector[N] eta = alpha + day_c * b_day + rows_dot_product(x, beta[ subj, ]);
    y_rep = to_vector(normal_rng(eta, sigma));
    for (n in 1:N) {
      log_lik[n] = normal_lpdf(y[n] | eta[n], sigma);
    }
  }
}
