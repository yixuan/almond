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
den_types = c("True Latent", "Estimated Latent")
cols = c("#F9812A", "#1EBE39")
legend_cols = c("#F9812A", "#A5E5AF")
pdf_u = function(u) dgamma(u + 2, shape = 2, scale = 1)
cdf_u = function(u) pgamma(u + 2, shape = 2, scale = 1)
gen_u = function(n) rgamma(n, shape = 2, scale = 1) - 2
xlim_u = c(-5, 8)

set.seed(123)
dat = read_npz("../results/glmm_example.npz")
load("../results/glmm_example_lme4.RData")

## Histogram
n = nrow(dat$u_vae)
u_glmm = u_lme4[, 1]
u_vae = dat$u_vae[, 1]
u_bc = dat$u_bc[, 1]
est_df = data.frame(
    u = c(u_glmm, u_vae, u_bc),
    method = factor(rep(methods, each = n), levels = methods),
    metric = "Marginal Distribution"
)

## True latent marginal density curve
u = seq(xlim_u[1], xlim_u[2], length.out = 300)
den = pdf_u(u)
u_df = data.frame(u = u, den = den, metric = "Marginal Distribution")

## Distance metrics
u0 = gen_u(n)
ks = sapply(list(u_glmm, u_vae, u_bc), function(x) ks.test(x, cdf_u)$statistic)
w = sapply(list(u_glmm, u_vae, u_bc), function(x) wasserstein1d(x, u0))
dist_df = data.frame(
    label = sprintf("D = %.4f\nW = %.4f", ks, w),
    method = factor(methods, levels = methods),
    metric = "Marginal Distribution"
)

## A "fake" data set to create the legend
legend_df = data.frame(
    x = rep(c(0, 0.1), 2),
    y = rep(c(0, 0.1), 2),
    type = rep(factor(den_types, levels = den_types), each = 2)
)

g = ggplot(est_df) +
    geom_histogram(aes(x = u, y = ..density..), fill = cols[2], alpha = 0.4, bins = 100) +
    geom_text(aes(label = label), data = dist_df,
              x = Inf, y = Inf, hjust = 1.2, vjust = 1.5, size = 4) +
    facet_grid(metric ~ method) +
    geom_line(aes(x = u, y = den), data = u_df,
              size = 0.8, color = cols[1]) +
    ylab("Density") +
    coord_cartesian(xlim = c(-5, 8)) +

    # Legend part
    geom_line(aes(x = x, y = y, color = type, size = type),
              alpha = 0, data = legend_df) +
    scale_color_manual("Density Type", values = legend_cols) +
    scale_size_manual("Density Type", values = c(1.2, 5)) +
    guides(color = guide_legend(keyheight = 1, keywidth = 3,
                                override.aes = list(alpha = 1))) +

    theme_bw(base_size = 20, base_family = "source") +
    theme(axis.title.x = element_blank(),
          legend.title = element_text(face = "bold"),
          legend.position = "bottom",
          legend.margin = margin(),
          legend.box.margin = margin(t = -15))

font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/glmm_example_1.pdf", g, width = 12, height = 4)

## Compute the lambda function of Copula
sourceCpp("copula_z.cpp")
lambdafun = function(x1, x2)
{
    z = copula_z(x1, x2)
    Fz = ecdf(z)
    function(x) x - Fz(x)
}
## Take a subset
subid = 1:10000
u_glmm1 = u_glmm[subid]
u_vae1 = u_vae[subid]
u_bc1 = u_bc[subid]
u_glmm2 = u_lme4[subid, 2]
u_vae2 = dat$u_vae[subid, 2]
u_bc2 = dat$u_bc[subid, 2]

lambda_cop = function(t) -0.5 * (t - t^3)
lambda_eb = lambdafun(u_glmm1, u_glmm2)
lambda_vae = lambdafun(u_vae1, u_vae2)
lambda_bc = lambdafun(u_bc1, u_bc2)

nt = 300
t = seq(0, 1, length.out = nt)
lambdat = lambda_cop(t)
den_types = c("True λ(t)", "Estimated λ(t)")

lambda_df = data.frame(
    t = rep(t, 6),
    lambda = c(lambdat, lambda_eb(t), lambdat, lambda_vae(t), lambdat, lambda_bc(t)),
    method = factor(rep(methods, each = nt * 2), levels = methods),
    type = factor(rep(rep(den_types, each = nt), 3), levels = den_types),
    metric = "Copula λ(t)"
)

# L1 Distance
l1dist = function(f) integrate(function(x) abs(f(x) - lambda_cop(x)), 0, 1, subdivisions = 400)$value
dist_df = data.frame(
    label = sprintf("L1 = %.4f", c(l1dist(lambda_eb), l1dist(lambda_vae), l1dist(lambda_bc))),
    method = factor(methods, levels = methods),
    metric = "Copula λ(t)"
)

g = ggplot(lambda_df) +
    geom_line(aes(x = t, y = lambda, color = type, linetype = type, group = type),
              size = 1) +
    geom_text(aes(label = label), data = dist_df,
              x = Inf, y = -Inf, hjust = 1.2, vjust = -1, size = 4) +
    facet_grid(metric ~ method) +
    scale_y_continuous("Function Value") +
    scale_color_manual("Curve Type", values = cols) +
    scale_linetype_manual("Curve Type", values = c("solid", "31")) +
    guides(color = guide_legend(keyheight = 1, keywidth = 3),
           linetype = guide_legend(override.aes = list(size = 1.2))) +
    theme_bw(base_size = 20, base_family = "source") +
    theme(axis.title.x = element_blank(),
          legend.title = element_text(face = "bold"),
          legend.position = "bottom",
          legend.margin = margin(),
          legend.box.margin = margin(t = -10))

font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/glmm_example_2.pdf", g, width = 12, height = 4)
