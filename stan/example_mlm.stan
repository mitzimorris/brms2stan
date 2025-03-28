/**
 * Varying slopes and intercept hierarchical linear regression.
 * N observations organized into J groups, with jj[n] being the group
 * and x[n, 1:K] the covariates for observation n.  The covariate
 * matrix x should include a column of 1s to include a slope.
 * 
 * The slopes and intercept per group have a multivariate normal prior
 * and the scale has an exponential prior.  The location of the
 * multivariate normal prior has a standard normal hyperprior and its
 * covariance is decomposed into a correlation matrix with an LKJ
 * hyperprior and a scale vector with an exponential hyperprior. In
 * symbols: 
 *
 * Likelihod:
 *   y[n] ~ normal(x[n] * beta[1:K, jj[n]], sigma) for n in 1:N
 *
 * Priors:
 *   sigma ~ exponential(1)
 *   beta[1:K, j] ~ multi_normal(nu, Sigma) for j in 1:J
 * 
 * Hyperpriors:
 *   nu ~ normal(0, 1)
 *   scale(Sigma) ~ exponential(1)
 *   corr(Sigma) ~ lkj(2)
 *
 * where scale(Sigma) is the scale vector and corr(Sigma) is the
 * correlation matrix of Sigma.
 *
 * For efficiency and numerical stability, the covariance and
 * correlation matrices are Cholesky factored.
 */
data {
  int<lower=0> J;                      // number of groups
  int<lower=0> N;                      // number of observations
  array[N] int<lower=1, upper=J> jj;   // group per observation
  int<lower=1> K;                      // number of covariates
  matrix[N, K] x;                      // data matrix
  vector[N] y;                         // observations
}
parameters {
  vector[K] nu;                        // location of beta[ , j]
  vector<lower=0>[K] tau;              // scale of beta[ , j]
  cholesky_factor_corr[K] L_Omega;     // Cholesky of correlation of beta[ , j]
  matrix[K, J] beta_std;               // standard beta (beta - nu) / Sigma
  real<lower=0> sigma;                 // observation error for y
}
transformed parameters {
  matrix[K, J] beta = rep_matrix(nu, J)
                      + diag_pre_multiply(tau, L_Omega) * beta_std;
}
model {
  nu ~ normal(0, 1);
  tau ~ exponential(1);
  L_Omega ~ lkj_corr_cholesky(2);
  to_vector(beta_std) ~ normal(0, 1);  // beta[ , j] ~ multi_normal(nu, Sigma)
  sigma ~ exponential(1);
  y ~ normal(rows_dot_product(x, beta[ , jj]'), sigma);
}
generated quantities {
  matrix[K, K] Sigma                   // covariance of beta[, j]
    = multiply_lower_tri_self_transpose(diag_pre_multiply(tau, L_Omega));
}
