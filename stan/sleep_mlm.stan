/**
 * Multilevel linear model for sleep study data
 * Formula: Reaction ~ 1 + Days + (1 + Days | Subject)
 *
 * This model includes:
 * - Global intercept and slope for Days
 * - Subject-specific random intercepts and slopes
 * - Correlation between random effects
 *
 *
 * Need to account for global effects and group-level effects
 *
 * Likelihod:
 *   y[n] ~ normal(x[n] * beta[1:K, jj[n]], sigma) for n in 1:N



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
  matrix[N, 2] x;
  x[ , 1] = rep_vector(1, N);
  x[ , 2] = day;
}
parameters {
  real b_intercept;  // global intercept
  real b_day; // global day effect
  real<lower=0> sigma;
  
  // Subject-level effects - non-centered parameterization
  vector[2] nu;                        // location of beta[ , j]
  vector<lower=0>[2] tau;              // scale of beta[ , j]
  cholesky_factor_corr[2] L_Omega;  // Cholesky factor of correlation matrix
  matrix[2, J] beta_std; // standardized random effects
}
transformed parameters {
  // random effects matrix scaled, transposed
  matrix[J, 2] beta = rep_matrix(nu, J)'
    + (diag_pre_multiply(tau, L_Omega) * beta_std)';
}
model {
  nu ~ std_normal();
  tau ~ exponential(1);
  L_Omega ~ lkj_corr_cholesky(2);
  b_intercept ~ normal(250, 50);  // weakly informative prior for intercept
  b_day ~ normal(10, 10);         // weakly informative prior for day effect
  to_vector(beta_std) ~ std_normal(); 
  sigma ~ exponential(1);
  
  vector[N] eta = b_intercept + b_day * day + rows_dot_product(x, beta[ subj, ]);
  y ~ normal(eta, sigma);
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
    vector[N] eta = b_intercept + b_day * day + rows_dot_product(x, beta[ subj, ]);
    y_rep = to_vector(normal_rng(eta, sigma));
    for (n in 1:N) {
      log_lik[n] = normal_lpdf(y[n] | eta[n], sigma);
    }
  }
}
