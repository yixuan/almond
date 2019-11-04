import time
import mxnet as mx
from mxnet.gluon import nn
import numpy as np

from almond import LatentModel, VAEEncoder, VAEDecoder, ConditionalDistr

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

        # log_pdf = -0.5 * F.square(linear - y)

        # linear = F.clip(linear, -20.0, 20.0)
        # log_pdf = (y - 1.0) * linear + F.log(F.sigmoid(linear))

        linear = F.clip(linear, -10.0, 10.0)
        log_pdf = y * linear - F.exp(linear) - F.gammaln(y + 1.0)
        return log_pdf

class Logger:
    def __init__(self, filename=None):
        self.filename = filename
        if filename is not None:
            with open(filename, "w") as f:
                f.write("")

    def log(self, str):
        if self.filename is not None:
            with open(self.filename, "a") as f:
                f.write(str)
                f.write("\n")
        else:
            print(str)


# Number of GPUs for computation
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
# Otherwise, use CPU
# ctx = [mx.cpu()]

# Logger
# logger = Logger("glmm_simulation.log")
logger = Logger()

# Data
dat = np.load("../results/glmm_rep_data.npz")
print(dat.files)
x = dat["x"]
z = dat["z"]
y = dat["y"]
beta = dat["beta"]

# Number of simulation runs
nsim = x.shape[0]
# Dimension of the observed data
n = x.shape[1]
px = x.shape[2]
pz = z.shape[2]

# Result
est_nsamp = 100000
beta_vae = np.zeros(shape=(nsim, px))
beta_bc = np.zeros(shape=(nsim, px))
u_vae = np.zeros(shape=(nsim, est_nsamp, 2))
u_bc = np.zeros(shape=(nsim, est_nsamp, 2))

# Model fitting
dim_all = px + pz + 1
batch_size = 1000

for i in range(nsim):
    logger.log("===> Simulation {}\n".format(i))
    np.random.seed(123)
    mx.random.seed(123)

    t1 = time.time()

    # Data
    xzy = mx.nd.array(np.hstack((x[i, :, :], z[i, :, :], y[i, :, :])))

    # Model
    model = LatentModel(GLMMLogLik(px, pz),
                        encoder=VAEEncoder([dim_all, 256, 128], latent_dim=pz, act="softrelu"),
                        decoder=VAEDecoder([128, 256, pz], latent_dim=pz, npar=1, act="softrelu"),
                        sim_z=10, nchain=30, ctx=ctx)
    model.init(lr=0.0001, lr_bc=0.0001)
    bhat = model.module.log_cond_pdf.fe.weight

    # Model fitting
    logger.log("     => VAE")

    model.fit(xzy, epochs=2000, batch_size=batch_size, eval_nll=False, verbose=False)
    mu_est_vae = model.simulate_prior(est_nsamp)[0]
    b_est_vae = bhat.data(ctx=ctx[0]).asnumpy()

    logger.log("     => Bias correction")

    particles = model.fit_bc(xzy, epochs=2000, warmups=2000, batch_size=batch_size,
                             burnin=20, step_size=0.01, eval_nll=False, verbose=False)
    mu_est_bc = model.simulate_prior(est_nsamp)[0]
    b_est_bc = bhat.data(ctx=ctx[0]).asnumpy()

    u_vae[i, :, :] = mu_est_vae[:, range(2)]
    u_bc[i, :, :] = mu_est_bc[:, range(2)]

    t2 = time.time()
    logger.log("===> Simulation {} finished in {} seconds\n".format(i, t2 - t1))

np.savez("../results/glmm_simulation.npz", beta_vae=beta_vae, beta_bc=beta_bc, u_vae=u_vae, u_bc=u_bc)
