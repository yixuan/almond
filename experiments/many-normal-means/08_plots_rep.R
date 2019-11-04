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
dists = c("K-S Distance", "1-Wasserstein")

meta = list(
    normal = list(
        name  = "normal",
        label = "Normal",
        cdf   = function(x) pnorm(x, 2, sqrt(0.5))
    ),
    exp = list(
        name  = "exp",
        label = "Exponential",
        cdf   = function(x) pexp(x, rate = 0.5)
    ),
    mix = list(
        name  = "mix",
        label = "Mixture of Normals",
        cdf   = function(x) 0.4 * pnorm(x, 0, 0.5) + 0.6 * pnorm(x, 3, 0.5)
    )
)

set.seed(123)
dist_df = vector("list", 3)
for(i in 1:3)
{
    distr = meta[[i]]
    dat = read_npz(sprintf("../results/mnn_%s_simulation.npz", distr$name))
    nsim = nrow(dat$mu0_dat)
    ks_eb = ks_vae = ks_bc = w_eb = w_vae = w_bc = numeric(nsim)
    for(j in seq_len(nsim))
    {
        ks_eb[j]  = ks.test(dat$mu_est_eb_dat[j, ],  distr$cdf)$statistic
        ks_vae[j] = ks.test(dat$mu_est_vae_dat[j, ], distr$cdf)$statistic
        ks_bc[j]  = ks.test(dat$mu_est_bc_dat[j, ],  distr$cdf)$statistic

        w_eb[j]   = wasserstein1d(dat$mu0_dat[j, ], dat$mu_est_eb_dat[j, ])
        w_vae[j]  = wasserstein1d(dat$mu0_dat[j, ], dat$mu_est_vae_dat[j, ])
        w_bc[j]   = wasserstein1d(dat$mu0_dat[j, ], dat$mu_est_bc_dat[j, ])
    }

    dist_df[[i]] = data.frame(
        x = c(ks_eb, w_eb, ks_vae, w_vae, ks_bc, w_bc),
        method = factor(rep(methods, each = nsim * 2), levels = methods),
        dist = factor(rep(rep(dists, each = nsim), 3), levels = dists),
        distr = distr$label
    )
}

dist_df = do.call(rbind, dist_df) %>%
    filter(!(distr == "Normal" & x > 0.2)) %>%
    filter(!(distr == "Exponential" & x > 1.0))

g = ggplot(dist_df) +
    geom_density(aes(x = x, color = method, linetype = method, fill = method, group = method),
                 size = 0.8, alpha = 0.1) +
    facet_grid(dist ~ distr, scales = "free") +
    scale_x_continuous(expand = expand_scale(mult = 0.07)) +
    ylab("Density") +
    scale_color_hue("Inference Method") +
    scale_fill_hue("Inference Method") +
    scale_linetype_manual("Inference Method", values = c("dotdash", "32", "solid")) +
    guides(color = guide_legend(keywidth = 3)) +
    theme_bw(base_size = 20, base_family = "source") +
    theme(axis.title.x = element_blank(),
          legend.title = element_text(face = "bold"),
          legend.position = "bottom",
          legend.margin = margin(),
          legend.box.margin = margin(t = -10))

font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/mnn_simulation.pdf", g, width = 12, height = 6)
