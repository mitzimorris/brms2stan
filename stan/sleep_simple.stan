data {
  int<lower=0> N;   vector[N] day;   vector[N] y;  // reaction time
}
transformed data {
  real day_mean = mean(day);
  vector[N] day_centered = day - day_mean;
}
parameters {
  real alpha;  real b_day;  // intercept, slope
  real<lower=0> sigma; // residual standard deviation
}
model {
  y ~ normal(alpha + day_centered * b_day, sigma);
  alpha ~ normal(250, 50);  // informed prior for human reaction times in ms
  b_day ~ normal(10, 10);  // weakly informed prior for per-day effect
  sigma ~ normal(0, 10);  // very weakly informative prior
}
generated quantities {
  real b_intercept = alpha - b_day * day_mean;
  array[N] real y_rep = normal_rng(alpha + day_centered * b_day, sigma);
}
