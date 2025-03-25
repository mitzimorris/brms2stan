library(ggplot2);
library(dplyr);
library(cmdstanr);
library(posterior);
options(mc.cores = 4);



nwarmup=5000;
nsampling=5000;

funnel_cp = cmdstan_model(stan_file="stan/neals_funnel_cp.stan")
fit_cp_10 = funnel_cp$sample(iter_warmup=nwarmup, iter_sampling=nsampling, refresh=2500);

funnel_nc = cmdstan_model(stan_file="stan/neals_funnel_nc.stan")
fit_nc_10 = funnel_nc$sample(iter_warmup=nwarmup, iter_sampling=nsampling, refresh=2500);



get_x1_y = function(x) {
  fit_rvars = as_draws_rvars(x$draws());
  y = draws_of(fit_rvars$y);
  x1 = draws_of(fit_rvars$x[1]);
  return(data.frame(y = y, x1 = x1));
}

get_fit_divergs = function(x) {
  divergs = vector("numeric");
  divergs = extract_variable(x$sampler_diagnostics(), "divergent__");
  d_idxs = which(divergs==1);
  fit_rvars = as_draws_rvars(x$draws());
  y_all = draws_of(fit_rvars$y);
  x1_all = as.vector(draws_of(fit_rvars$x[1]));
  return(data.frame(y=y_all[d_idxs],
                    x1=x1_all[d_idxs]));
}

cp_fit_y_x1 = get_x1_y(fit_cp_10);
cp_divs = get_fit_divergs(fit_cp_10);

p1 = ggplot(cp_fit_y_x1, aes(x=x1, y=y));
p1 = p1 + geom_point(colour="black", size=0.55, alpha=0.1);
p1 = p1 + geom_point(data=cp_divs, aes(x=x1, y=y), colour="darkorange", size=0.8);
p1 = p1 + labs(title="centered parameterization",
               subtitle="20000 draws, divergences in orange", x = "x[1]", y = "y");
p1 = p1 + theme(aspect.ratio = 5/4);
ggsave("img/funnel_fit_cp.png", plot=p1, width=6, height=8);

nc_fit_y_x1 = get_x1_y(fit_nc_10);
nc_divs = get_fit_divergs(fit_nc_10);

p1 = ggplot(nc_fit_y_x1, aes(x=x1, y=y));
p1 = p1 + geom_point(colour="black", size=0.55, alpha=0.1);
p1 = p1 + geom_point(data=nc_divs, aes(x=x1, y=y), colour="darkorange", size=0.8);
p1 = p1 + labs(title="non-centered parameterization",
               subtitle="20000 draws, (no divergences)", x = "x[1]", y = "y");
p1 = p1 + theme(aspect.ratio = 5/4);
ggsave("img/funnel_fit_nc.png", plot=p1, width=6, height=8);
