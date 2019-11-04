library(copula)
library(ggplot2)
library(showtext)
library(reticulate)
use_python("/usr/bin/python3")
py_config()
np = import("numpy")

# Dimension
p = 5
nsim = 10000

## Clayton copula with theta=2
set.seed(123)
cop = claytonCopula(param = 2, dim = p)
u = rCopula(nsim, cop)

## Gamma marginal distribution
mpdf = function(x) dgamma(x + 2, shape = 2, scale = 1)
mcdf = function(x) pgamma(x + 2, shape = 2, scale = 1)
micdf = function(u) qgamma(u, shape = 2, scale = 1) - 2
u = micdf(u)

## Marginal density plot
plot(density(u[, 1]), ylim = c(0, 0.4))
curve(mpdf(x), col = "red", add = TRUE, n = 401)

## Scatterplot
nsub = 1000
gdat = data.frame(u1 = u[1:nsub, 1], u2 = u[1:nsub, 2])
g = ggplot(gdat, aes(x = u1, y = u2)) +
    geom_point(size = 3, alpha = 0.1) +
    xlab("Dimension 1") + ylab("Dimension 2") +
    coord_fixed(xlim = c(-3, 8), ylim = c(-3, 8), expand = FALSE) +
    theme_bw(base_size = 20, base_family = "source")
font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/glmm_latent_scatter.pdf", g, width = 6, height = 6)

## Density contour
cop = claytonCopula(param = 2, dim = 2)
ngrid = 200
u0 = seq(-3, 3, length.out = ngrid)
denu = data.frame(
    u1 = rep(u0, ngrid),
    u2 = rep(u0, each = ngrid),
    den = as.numeric(ngrid^2)
)
denu$den = mpdf(denu$u1) * mpdf(denu$u2) * dCopula(mcdf(cbind(denu$u1, denu$u2)), cop)
g = ggplot(denu, aes(x = u1, y = u2)) +
    geom_raster(aes(fill = den)) +
    geom_contour(aes(z = den), color = "white", size = 0.1, alpha = 0.3, bins = 50) +
    scale_fill_distiller("Density", palette = "Spectral", direction = -1) +
    guides(fill = guide_colorbar(barheight = 20, order = 1)) +
    scale_x_continuous("Dimension 1", breaks = seq(-2.5, 2.5, by = 1)) +
    scale_y_continuous("Dimension 2", breaks = seq(-2.5, 2.5, by = 1)) +
    coord_fixed(xlim = c(-3, 3), ylim = c(-3, 3), expand = FALSE) +
    theme_bw(base_size = 20, base_family = "source")
font_add_google("Source Sans Pro", "source")
showtext_auto()
ggsave("../results/glmm_latent_contour.pdf", g, width = 7.5, height = 6)

## Simulate GLMM data
set.seed(123)
x = 0.2 * matrix(rnorm(nsim * p), nsim, p)
z = 0.2 * matrix(rnorm(nsim * p), nsim, p)
beta = runif(p, -2, 2)
linear = c(x %*% beta) + rowSums(z * u)
y = matrix(rpois(nsim, exp(linear)) + 0.0, ncol = 1)
np$savez_compressed("../results/glmm_data.npz", x = x, z = z, y = y, beta = beta)



# Replications
nrep = 100

set.seed(123)
cop = claytonCopula(param = 2, dim = p)
u = rCopula(nsim * nrep, cop)
u = micdf(u)
u = array(u, dim = c(nrep, nsim, p))

x = 0.2 * array(rnorm(nrep * nsim * p), dim = c(nrep, nsim, p))
z = 0.2 * array(rnorm(nrep * nsim * p), dim = c(nrep, nsim, p))
beta = matrix(runif(nrep * p, -2, 2), nrep, p)
y = array(0, dim = c(nrep, nsim, 1))
for(i in 1:nrep)
{
    print(i)
    linear = c(x[i, , ] %*% beta[i, ]) + rowSums(z[i, , ] * u[i, , ])
    y[i, , ] = rpois(nsim, exp(linear)) + 0.0
}
np$savez_compressed("../results/glmm_rep_data.npz", x = x, z = z, y = y, beta = beta)
