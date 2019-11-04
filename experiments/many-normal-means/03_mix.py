import math
import mxnet as mx
import numpy as np
from scipy.stats import norm

from almond import LatentModel, VAEEncoder, VAEDecoder, ConditionalNormal

# Number of GPUs for computation
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
# Otherwise, use CPU
# ctx = [mx.cpu()]

# Generate data
def gen_mu_mixture_prior(n, mu1=0.0, mu2=3.0, p1=0.3, sigma=1.0):
    mean = np.zeros(n) + mu2
    ind = np.random.binomial(1, p1, n).astype(bool)
    mean[ind] = mu1
    mu = np.random.normal(loc=mean, scale=sigma)
    x = mu + np.random.randn(n)
    return mu, x

# Data
np.random.seed(123)
mx.random.seed(123)
n = 1000
mix_par = {"mu1": 0.0, "mu2": 3.0, "p1": 0.4, "sigma": 0.5}
norm1 = norm(loc=mix_par["mu1"], scale=mix_par["sigma"])
norm2 = norm(loc=mix_par["mu2"], scale=mix_par["sigma"])
def true_dist(x):
    return mix_par["p1"] * norm1.cdf(x) + (1.0 - mix_par["p1"]) * norm2.cdf(x)
def true_pdf(x):
    return mix_par["p1"] * norm1.pdf(x) + (1.0 - mix_par["p1"]) * norm2.pdf(x)

mu, x = gen_mu_mixture_prior(n, **mix_par)
xt = mx.nd.array(x).reshape(-1, 1)

# Model
model = LatentModel(ConditionalNormal(dimu=1),
                    encoder=VAEEncoder([1, 50, 100, 50], latent_dim=1),
                    decoder=VAEDecoder([50, 100, 50, 1], latent_dim=1, npar=1),
                    sim_z=100, nchain=100, ctx=ctx)
model.init(lr=0.01, lr_bc=0.01)

# Model fitting
batch_size = n         # mini-batch size
est_nsamp = 10000      # size of Monte Carlo sample to approximate the density
epochs_vae = 1000      # pre-train model using VAE
epochs_bc = 1000       # bias correction step

# VAE
model.fit(xt, epochs=epochs_vae, batch_size=batch_size)
mu_est_vae = model.simulate_prior(est_nsamp)[0].squeeze()

# ALMOND bias correction
particles = model.fit_bc(xt, epochs=epochs_bc, warmups=100, batch_size=batch_size, burnin=10, step_size=0.01)
mu_est_bc = model.simulate_prior(est_nsamp)[0].squeeze()

# Empirical Bayes estimation
eb_mu = np.mean(x)
eb_var = np.var(x) - 1.0
mu_est_eb = np.random.normal(eb_mu, math.sqrt(eb_var), est_nsamp)

np.savez("../results/mnn_mix_example.npz", vae=mu_est_vae, bc=mu_est_bc, eb=mu_est_eb)
