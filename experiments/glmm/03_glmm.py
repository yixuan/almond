import mxnet as mx
from mxnet.gluon import nn
import numpy as np
from scipy.stats import gamma

from almond import LatentModel, VAEEncoder, VAEDecoder, ConditionalDistr

# Number of GPUs for computation
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
# Otherwise, use CPU
# ctx = [mx.cpu()]

# Data
np.random.seed(123)
mx.random.seed(123)

u_dist = gamma(a=2, loc=-2, scale=1)
u_pdf = u_dist.pdf
u_cdf = u_dist.cdf
u_rvs = u_dist.rvs

dat = np.load("../results/glmm_data.npz")
print(dat.files)
x = dat["x"]
z = dat["z"]
y = dat["y"]
beta = dat["beta"]
n = x.shape[0]
px = x.shape[1]
pz = z.shape[1]

xzy = mx.nd.array(np.hstack((x, z, y)))

# Conditional distribution
class GLMMLogLik(ConditionalDistr):
    # Constructor
    def __init__(self, px, pz):
        super(GLMMLogLik, self).__init__()

        # Dimension of X
        self.px = px
        # Dimension of Z
        self.pz = pz
        # Fixed effects
        self.fe = nn.Dense(1, use_bias=False, in_units=px)

    # Input: xzy [N x (px + pz + 1)], u [ [N x pz] ]
    # Output size [N]
    # Argument x => xzy
    def hybrid_forward(self, F, x, u):
        xzy = x
        x = F.slice(xzy, begin=(None, 0), end=(None, px))
        z = F.slice(xzy, begin=(None, px), end=(None, px + pz))
        y = F.slice(xzy, begin=(None, px + pz), end=(None, px + pz + 1)).squeeze()

        linear = self.fe(x).squeeze() + F.sum(z * u[0], axis=1)
        linear = F.clip(linear, -10.0, 10.0)
        log_pdf = y * linear - F.exp(linear) - F.gammaln(y + 1.0)
        return log_pdf

# Model
np.random.seed(123)
mx.random.seed(123)
dim_all = px + pz + 1
model = LatentModel(GLMMLogLik(px, pz),
                    encoder=VAEEncoder([dim_all, 256, 128], latent_dim=pz, act="softrelu"),
                    decoder=VAEDecoder([128, 256, pz], latent_dim=pz, npar=1, act="softrelu"),
                    sim_z=10, nchain=30, ctx=ctx)
model.init(lr=0.0001, lr_bc=0.0001)

# Model fitting
batch_size = 1000
est_nsamp = 100000
bhat = model.module.log_cond_pdf.fe.weight

# VAE
model.fit(xzy, epochs=2000, batch_size=batch_size)
mu_est_vae = model.simulate_prior(est_nsamp)[0]
b_est_vae = bhat.data(ctx=ctx[0]).asnumpy()

# ALMOND bias correction
particles = model.fit_bc(xzy, epochs=2000, warmups=2000, batch_size=batch_size, burnin=20, step_size=0.01)
mu_est_bc = model.simulate_prior(est_nsamp)[0]
b_est_bc = bhat.data(ctx=ctx[0]).asnumpy()

np.savez("../results/glmm_example.npz", u_vae=mu_est_vae, b_vae=b_est_vae, u_bc=mu_est_bc, b_bc=b_est_bc)
