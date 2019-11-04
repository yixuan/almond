library(lme4)
library(mvtnorm)
library(reticulate)
use_python("/usr/bin/python3")
py_config()
np = import("numpy")

# Read data
read_npz = function(npz_file)
{
    npz = np$load(npz_file)
    files = npz$files
    nfiles = length(files)

    res = vector("list", nfiles)
    for(i in seq_along(files))
    {
        res[[i]] = npz$f[[files[i]]]
    }
    names(res) = files

    res
}

simdat = read_npz("../results/glmm_data.npz")
x = simdat$x
z = simdat$z
y = simdat$y

colnames(x) = sprintf("x%d", 1:ncol(x))
colnames(z) = sprintf("z%d", 1:ncol(z))
dat = cbind(as.data.frame(x), as.data.frame(z))
dat$y = as.numeric(y)
dat$id = 1:nrow(dat)

# Fit GLMM using lme4
set.seed(123)
mod = glmer(y ~ 0 + x1 + x2 + x3 + x4 + x5 +
                (0 + z1 + z2 + z3 + z4 + z5 | id),
            data = dat, family = poisson(),
            control = glmerControl(check.nobs.vs.nlev = "ignore",
                                   check.nobs.vs.rankZ = "ignore",
                                   check.nobs.vs.nRE = "ignore",
                                   calc.derivs = FALSE,
                                   optCtrl = list(maxfun = 1000)),
            verbose = 2)

# Extract covariance matrix
rcov = function(mod)
{
    v = VarCorr(mod)
    r = attr(v[[1]], "correlation")
    diag(r) = attr(v[[1]], "stddev")
    sdcor2cov(r)
}

# Generate Monte Carlo sample of random effects
set.seed(123)
beta_lme4 = fixef(mod)
Sigma = rcov(mod)

nsim = 100000
u_lme4 = rmvnorm(nsim, sigma = Sigma[1:2, 1:2])

save(beta_lme4, u_lme4, file = "../results/glmm_example_lme4.RData")
