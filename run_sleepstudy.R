setwd("/Users/mitzi/github/zmorris/workshops/brms2stan")
suppressMessages(library(brms))
suppressMessages(library(lme4))
suppressMessages(library(cmdstanr))
suppressMessages(library(posterior))

options(digits=3)
options(width = 120)
options(scipen=20)

options(mc.cores=4)

# Assemble sleepstudy data as list for Stan
sleep_data = list(
    N = nrow(sleepstudy),
    J = length(unique(sleepstudy$Subject)),
    subj = as.integer(sleepstudy$Subject),
    day = sleepstudy$Days,
    y = as.double(sleepstudy$Reaction))

# Stan complete pooling model
sleep_simple = cmdstan_model("stan/sleep_simple.stan")
sleep_simple_stanfit = sleep_simple$sample(data = sleep_data)
as.data.frame(sleep_simple_stanfit$summary(variables = c('b_intercept', 'b_day', 'sigma')))


# BRMS model - complete pooling
# match Stan priors
priors <- c(
set_prior("normal(250, 50)", class = "Intercept"),
set_prior("normal(10, 10)", class = "b"),            # fixed slopes
set_prior("normal(0, 10)", class = "sigma")
)

sleep_simple_brmsfit = brm(Reaction ~ Days, data = sleepstudy, prior = priors)
sleep_simple_brmsfit

# Stan multilevel model
sleep_mlm = cmdstan_model(stan_file = "stan/sleep_mlm.stan")
sleep_mlm_stanfit = sleep_mlm$sample(data = sleep_data)
as.data.frame(sleep_mlm_stanfit$summary(variables = c('b_intercept', 'b_day', 'sigma', 'sd_intercept', 'sd_day', 'cor_intercept_day')))


# BRMS multilevel model
sleep_mlm_brmsfit <- brm(Reaction ~ Days + (Days|Subject),  data = sleepstudy, prior = priors)
sleep_mlm_brmsfit



# plot posterior predictive - TODO
simple_rvars = as_draws_rvars(sleep_simple_stanfit$draws())
y_rep = draws_of(simple_rvars$y_rep)
