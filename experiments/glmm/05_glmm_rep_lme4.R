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

simdat = read_npz("../results/glmm_rep_data.npz")
x = simdat$x
z = simdat$z
y = simdat$y
beta = simdat$beta

# Results
nrep = dim(x)[1]
px = dim(x)[3]
pz = dim(z)[3]
nsim = 100000

beta_lme4 = matrix(0, nrep, px)
u_lme4 = array(0, dim = c(nrep, nsim, 2))

# Extract covariance matrix
rcov = function(mod)
{
    v = VarCorr(mod)
    r = attr(v[[1]], "correlation")
    diag(r) = attr(v[[1]], "stddev")
    sdcor2cov(r)
}

# Computing
for(i in 1:nrep)
{
    print(i)
    xi = x[i, , ]
    zi = z[i, , ]
    yi = y[i, , ]

    colnames(xi) = sprintf("x%d", 1:px)
    colnames(zi) = sprintf("z%d", 1:pz)
    dat = cbind(as.data.frame(xi), as.data.frame(zi))
    dat$y = as.numeric(yi)
    dat$id = 1:nrow(dat)

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

    beta_lme4[i, ] = fixef(mod)
    Sigma = rcov(mod)
    u_lme4[i, , ] = rmvnorm(nsim, sigma = Sigma[1:2, 1:2])
}

save(beta_lme4, u_lme4, file = "../results/glmm_simulation_lme4.RData")
