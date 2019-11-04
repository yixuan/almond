library(transport)
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



methods = c("Empirical Bayes", "VAE", "ALMOND")

meta = list(
    normal = list(
        name  = "normal",
        label = "Normal",
        xlim  = c(-2, 6),
        text  = c(4, 0.6),
        pdf   = function(x) dnorm(x, 2, sqrt(0.5)),
        cdf   = function(x) pnorm(x, 2, sqrt(0.5)),
        gen_u = function(n) rnorm(n, 2, sqrt(0.5)),
        gen_x = function(n) rnorm(n, 2, sqrt(0.5)) + rnorm(n)
    ),
    exp = list(
        name  = "exp",
        label = "Exponential",
        xlim  = c(-5, 15),
        text  = c(5, 0.6),
        pdf   = function(x) dexp(x, rate = 0.5),
        cdf   = function(x) pexp(x, rate = 0.5),
        gen_u = function(n) rexp(n, rate = 0.5),
        gen_x = function(n) rexp(n, rate = 0.5) + rnorm(n)
    ),
    mix = list(
        name  = "mix",
        label = "Mixture of Normals",
        xlim  = c(-5, 8),
        text  = c(5, 0.6),
        pdf   = function(x) 0.4 * dnorm(x, 0, 0.5) + 0.6 * dnorm(x, 3, 0.5),
        cdf   = function(x) 0.4 * pnorm(x, 0, 0.5) + 0.6 * pnorm(x, 3, 0.5),
        gen_u = function(n)
        {
            mean = rep(3, n)
            ind = rbinom(n, 1, 0.4)
            mean[as.logical(ind)] = 0
            rnorm(n, mean, 0.5)
        },
        gen_x = function(n)
        {
            mean = rep(3, n)
            ind = rbinom(n, 1, 0.4)
            mean[as.logical(ind)] = 0
            mu = rnorm(n, mean, 0.5)
            mu + rnorm(n)
        }
    )
)

set.seed(123)
x_df = mu_df = est_df = dist_df = vector("list", 3)
for(i in 1:3)
{
    ## Histogram
    distr = meta[[i]]
    dat = read_npz(sprintf("../results/mnn_%s_example.npz", distr$name))
    est_df[[i]] = data.frame(
        mu = c(dat$eb, dat$vae, dat$bc),
        method = factor(rep(methods, each = length(dat$eb)), levels = methods),
        distr = distr$label
    )

    ## Data density curve
    den = density(distr$gen_x(10000), adjust = 1.5, from = distr$xlim[1], to = distr$xlim[2])
    x_df[[i]] = data.frame(mu = den$x, den = den$y, distr = distr$label)

    ## True latent density curve
    mu = seq(distr$xlim[1], distr$xlim[2], length.out = 300)
    den = distr$pdf(mu)
    mu_df[[i]] = data.frame(mu = mu, den = den, distr = distr$label)

    ## Distance measures
    mu0 = distr$gen_u(10000)
    ks = sapply(list(dat$eb, dat$vae, dat$bc),
                function(dat) ks.test(dat, distr$cdf)$statistic)
    w = sapply(list(dat$eb, dat$vae, dat$bc),
               function(dat) wasserstein1d(dat, mu0))
    dist_df[[i]] = data.frame(
        x = rep(distr$text[1], 3),
        y = rep(distr$text[2], 3),
        label = sprintf("D = %.4f\nW = %.4f", ks, w),
        method = factor(methods, levels = methods),
        distr = distr$label
    )
}

est_df = do.call(rbind, est_df)
x_df = do.call(rbind, x_df)
mu_df = do.call(rbind, mu_df)
dist_df = do.call(rbind, dist_df)

# A "fake" data set to create the legend
types = c("Data", "True Latent", "Estimated Latent")
cols = c("steelblue", "#F9812A", "#1EBE39")
legend_cols = c("steelblue", "#F9812A", "#A5E5AF")
legend_df = data.frame(
    x = rep(c(0, 0.1), 3),
    y = rep(c(0, 0.1), 3),
    type = rep(factor(types, levels = types), each = 2)
)

g = ggplot(est_df) +
    geom_histogram(aes(x = mu, y = ..density..), fill = cols[3], alpha = 0.4, bins = 100) +
    geom_text(aes(label = label), data = dist_df,
              x = -Inf, y = Inf, hjust = -0.2, vjust = 1.5, size = 4) +
    facet_grid(method ~ distr, scales = "free") +
    geom_line(aes(x = mu, y = den), data = x_df,
              size = 1, linetype = "32", color = cols[1]) +
    geom_line(aes(x = mu, y = den), data = mu_df,
              size = 0.8, color = cols[2]) +
    ylab("Density") +

    # Legend part
    geom_line(aes(x = x, y = y, color = type, linetype = type, size = type),
              alpha = 0, data = legend_df) +
    scale_color_manual("Density Type", values = legend_cols) +
    scale_linetype_manual("Density Type", values = c("32", "solid", "solid")) +
    scale_size_manual("Density Type", values = c(1.2, 1.2, 5)) +
    guides(color = guide_legend(keyheight = 1, keywidth = 3,
                                override.aes = list(alpha = 1))) +

    theme_bw(base_size = 20, base_family = "source") +
    theme(axis.title.x = element_blank(),
          legend.title = element_text(face = "bold"),
          legend.position = "bottom",
          legend.margin = margin(),
          legend.box.margin = margin(t = -10))

font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/mnn_example.pdf", g, width = 12, height = 9)
