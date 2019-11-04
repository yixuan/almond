library(transport)
library(copula)
library(Rcpp)
library(dplyr)
library(showtext)
library(ggplot2)
library(scales)
library(reticulate)
use_python("/usr/bin/python3")
py_config()
np = import("numpy")

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

methods = c("GLMM", "VAE", "ALMOND")
metrics = c("K-S Distance to F(u)",
            "1-Wasserstein Distance to F(u)",
            "L1 distance to λ(t)",
            "Fixed Effects Error")
mcdf = function(x) pgamma(x + 2, shape = 2, scale = 1)

set.seed(123)
simdat = read_npz("../results/glmm_rep_data.npz")
beta = simdat$beta

dat = read_npz("../results/glmm_simulation.npz")
load("../results/glmm_simulation_lme4.RData")
u_vae = dat$u_vae
u_bc = dat$u_bc
beta_vae = dat$beta_vae
beta_bc = dat$beta_bc
nsim = nrow(beta_lme4)

## Fixed effects
beta_err_lme4 = sqrt(rowSums((beta_lme4 - beta)^2))
beta_err_vae = sqrt(rowSums((beta_vae - beta)^2))
beta_err_bc = sqrt(rowSums((beta_bc - beta)^2))

## Compute the lambda function of Copula
sourceCpp("copula_z.cpp")
lambdafun = function(x1, x2)
{
    z = copula_z(x1, x2)
    Fz = ecdf(z)
    function(x) x - Fz(x)
}
lambda_cop = function(t) -0.5 * (t - t^3)

# L1 Distance
l1dist = function(f)
    integrate(function(x) abs(f(x) - lambda_cop(x)), 0, 1, subdivisions = 1000)$value

# Evaluation metrics
ks_lme4 = ks_vae = ks_bc = w_lme4 = w_vae = w_bc = l1_lme4 = l1_vae = l1_bc = numeric(nsim)
for(i in 1:nsim)
{
    print(i)

    ks_lme4[i] = ks.test(u_lme4[i, , 1], mcdf)$statistic
    ks_vae[i] = ks.test(u_vae[i, , 1], mcdf)$statistic
    ks_bc[i] = ks.test(u_bc[i, , 1], mcdf)$statistic

    mu0 = rgamma(1e5, shape = 2, scale = 1) - 2
    w_lme4[i] = wasserstein1d(u_lme4[i, , 1], mu0)
    w_vae[i] = wasserstein1d(u_vae[i, , 1], mu0)
    w_bc[i] = wasserstein1d(u_bc[i, , 1], mu0)

    ## Take a subset
    subid = 1:10000
    lambda_lme4 = lambdafun(u_lme4[i, subid, 1], u_lme4[i, subid, 2])
    lambda_vae = lambdafun(u_vae[i, subid, 1], u_vae[i, subid, 2])
    lambda_bc = lambdafun(u_bc[i, subid, 1], u_bc[i, subid, 2])

    l1_lme4[i] = l1dist(lambda_lme4)
    l1_vae[i] = l1dist(lambda_vae)
    l1_bc[i] = l1dist(lambda_bc)
}

gdat = data.frame(
    val = c(ks_lme4, ks_vae, ks_bc,
            w_lme4, w_vae, w_bc,
            l1_lme4, l1_vae, l1_bc,
            beta_err_lme4, beta_err_vae, beta_err_bc),
    method = factor(rep(rep(methods, each = nsim), 4), levels = methods),
    metric = factor(rep(metrics, each = nsim * 3), levels = metrics)
)

g = ggplot(gdat) +
    geom_density(aes(x = val, color = method, linetype = method, fill = method, group = method),
                 size = 0.8, alpha = 0.1) +
    facet_wrap(. ~ metric, nrow = 2, scales = "free") +
    ylab("Density") +
    scale_color_hue("Inference Method") +
    scale_fill_hue("Inference Method") +
    scale_linetype_manual("Inference Method", values = c("dotdash", "32", "solid")) +
    guides(color = guide_legend(keywidth = 3)) +
    theme_bw(base_size = 20, base_family = "source") +
    theme(axis.title.x = element_blank(),
          legend.title = element_text(face = "bold"),
          legend.position = "bottom")

font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/glmm_simulation.pdf", g, width = 10, height = 7)

