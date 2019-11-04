import math
import time

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import utils
from mxnet import autograd

from .utils import *
from .blocks import *
from .langevin import LangevinMultiChain

# A latent variable model
# X_i|u_i ~ f(x_i|u_i;theta)
# u = f(z), z ~ N(0, I)
# VAE part: use q(z|x) ~ N(m(x), v(x)) to approximate p(z|x)
class LatentModelModule(gluon.HybridBlock):
    # Constructor
    def __init__(self, log_cond_pdf, encoder, decoder, sim_z=10, nchain=10, ctx=[mx.cpu()]):
        super(LatentModelModule, self).__init__()

        # Device
        self.ctx = ctx

        # log-pdf of X_i given u_i
        self.log_cond_pdf = log_cond_pdf

        # Number of variables in the data
        self.input_dim = encoder.input_dim

        # Dimension of latent space
        self.latent_dim = encoder.latent_dim

        # Encoder
        self.enc = encoder

        # Number of z to simulate
        self.sim_z = sim_z

        # Number of chains for Langevin diffusion
        self.nchain = nchain

        # Decoder
        self.dec = decoder

        # Log-likelihood value
        self.loglik = VAEDecoderLogLik(decoder, sim_z, log_cond_pdf)
        self.loglik_params = self.loglik.collect_params()

        # Langevin log-pdf
        self.langevin = LangevinLogPDF(decoder, nchain, log_cond_pdf)
        self.langevin_loss = LangevinLoss(decoder, nchain, log_cond_pdf)

    # Input: m(x) [n x d], log[v(x)] [n x d]
    # Output: z ~ q(z|x), [(nz x n) x d]
    def simulate_latent(self, mu, logvar, nz, F):
        epsilon = F.random.normal(loc=F.zeros_like(mu), scale=F.ones_like(logvar), shape=nz).transpose((2, 0, 1))
        mu = mu.expand_dims(axis=0)
        sigma = logvar.__mul__(0.5).exp().expand_dims(axis=0)
        z = F.broadcast_add(mu, F.broadcast_mul(epsilon, sigma))
        return z.reshape((-1, self.latent_dim)), nz

    # -E_q(z|x) [log p(x|z)]
    def loss_likelihood(self, z, x, F):
        loglik = self.loglik(z, x)
        loss = -loglik / self.sim_z
        return loss

    # KL(q(z|x) || p(z))
    def loss_kl(self, mu, logvar, F):
        return -0.5 * F.sum(1.0 + logvar - F.square(mu) - logvar.exp())

    # Input x and output loss
    def hybrid_forward(self, F, x):
        mu, logvar = self.enc(x)
        z, _ = self.simulate_latent(mu, logvar, self.sim_z, F=F)

        loss_ll = self.loss_likelihood(z, x, F=F)
        loss_kl = self.loss_kl(mu, logvar, F=F)
        return loss_ll, loss_kl

    ######################## Sample latent variables ########################

    # Output: theta = f(z), z ~ p(z) ~ N(0, I)
    # Output numpy array [nsim x p] for each element in theta
    def simulate_theta_prior(self, nsim):
        epsilon = nd.random_normal(shape=(nsim, self.latent_dim), ctx=self.ctx[0])
        theta = self.dec(epsilon)
        return [e.asnumpy() for e in theta]

    # Input numpy array x [n x p]
    # Output: theta = f(zx), zx ~ q(z|x) ~ N(m(x), v(x))
    # Output numpy array [n x nsim x p]
    def simulate_theta_posterior(self, x, nsim):
        mu, logvar = self.enc(x)
        z, nz = self.simulate_latent(mu, logvar, nsim, F=nd)
        theta = self.dec(z)
        res = [e.reshape((nz, x.shape[0], -1)).transpose(axes=(1, 0, 2)).asnumpy() for e in theta]
        return res

    ######################## Evaluate likelihood ########################

    # Compute log[p(x)] using importance sampling
    # p(x) ~= 1/K * sum[p(z_i) * p(x|z_i) / q(z_i|x)], z_i ~ q(z|x)
    # log[p(x)] ~= log_sum_exp(v) - log(K)
    # v_i = log[p(z_i)] + log[p(x|z_i)] - log[q(z_i|x)]
    def negloglikelihood(self, x, nz=100):
        x = x.as_in_context(self.ctx[0])
        mu, logvar = self.enc(x)
        z, nz = self.simulate_latent(mu, logvar, nz, F=nd)
        fz = self.dec(z)

        mu_rep = rep_rows(mu, nz)
        logvar_rep = rep_rows(logvar, nz)
        x_rep = rep_rows(x, nz)

        # mu_rep, logvar_rep, and z are [(nz x n) x d]
        # x_rep and fz are [(nz x n) x p]

        const = 0.5 * self.latent_dim * math.log(2.0 * math.pi)
        log_pz = -0.5 * nd.sum(z * z, axis=1) - const  # row sum

        z_mux = z - mu_rep
        log_qzx = -0.5 * nd.sum(nd.square(z_mux) / logvar_rep.exp() + logvar_rep, axis=1) - const

        log_pxz = self.log_cond_pdf(x_rep, fz)

        log_w = (log_pz + log_pxz - log_qzx).reshape((nz, -1))
        log_px = log_sum_exp(log_w, axis=0, F=nd) - math.log(nz)
        return -log_px.mean().asscalar()

    ######################## MCMC part ########################

    # log[p(x, z)] = log[p(z)] + log[p(x|z)]
    # z [nchain x n x d], x [n x p]
    def log_pdf_grad(self, z, args):
        x = args["x"]
        z = z.copy()
        z.attach_grad()

        with autograd.record():
            log_pdf = self.langevin(z, x)

        log_pdf.backward()
        return z.grad

    # Sample from posterior using MCMC
    # x [n x p], init [nsim x n x d] or None
    # Output size [nsim x n x d]
    def simulate_z_posterior_mcmc(self, x, nsim, init=None, burnin=100, step_size=0.1):
        # This function relies on the hybridized self.langevin, so we need to make sure
        # nsim is equal to self.langevin.nchain. If not, we reinitialize self.langevin
        if nsim != self.langevin.nchain:
            self.langevin = LangevinLogPDF(self.dec, nsim, self.log_cond_pdf)
            self.langevin_loss = LangevinLoss(self.dec, nsim, self.log_cond_pdf)
            self.langevin.hybridize()
            self.langevin_loss.hybridize()

        n = x.shape[0]

        # For now neural network parameters do not need gradient
        for name, param in self.loglik_params.items():
            param.grad_req = "null"

        if init is None:
            mu, logvar = self.enc(x.as_in_context(self.ctx[0]))
            init, _ = self.simulate_latent(mu, logvar, nsim, F=nd)
            init = init.reshape((nsim, n, self.latent_dim))

        # init [nsim x n x d], x [n x d]
        # split_and_load() works on the first axis, first transpose init, and then transform back
        x_gpus = utils.split_and_load(x, self.ctx)
        init_gpus = utils.split_and_load(init.transpose((1, 0, 2)), self.ctx)
        chains_gpus = []
        for x, init in zip(x_gpus, init_gpus):
            # Transpose back
            init = init.transpose((1, 0, 2))
            sampler = LangevinMultiChain(shape=init.shape[1:], nchain=nsim, ctx=x.context)
            chains = sampler.sample(model=self, start=init, step_size=step_size, burnin=burnin, args={"x": x})
            chains_gpus.append(chains)

        nd.waitall()
        chains = [c.as_in_context(x.context) for c in chains_gpus]

        # Restore gradient flag
        for name, param in self.loglik_params.items():
            # Parameters that have no gradient, e.g. in batch normalization,
            # are unaffected
            param.grad_req = "write"

        return nd.concat(*chains, dim=1)

    # Bias-correction for loss function
    def bias_correction(self, x, init=None, burnin=100, step_size=0.1):
        postz = self.simulate_z_posterior_mcmc(x, nsim=self.nchain, init=init, burnin=burnin, step_size=step_size)

        # z [nsim x n x d], x [n x d]
        # split_and_load() works on the first axis, first transpose z, and then transform back
        x_gpus = utils.split_and_load(x, self.ctx)
        z_gpus = utils.split_and_load(postz.transpose((1, 0, 2)), self.ctx)

        loss_gpus = []
        with autograd.record():
            for x, z in zip(x_gpus, z_gpus):
                # Transpose back
                z = z.transpose((1, 0, 2))
                loss = self.langevin_loss(z.reshape((-1, self.latent_dim)), x)
                loss_gpus.append(loss)

        return loss_gpus, postz


# Train the VAE model
class LatentModel:
    # Constructor
    def __init__(self, log_cond_pdf, encoder, decoder, sim_z=10, nchain=10, ctx=[mx.cpu()]):
        # Device
        self.ctx = ctx

        # VAE module
        self.module = LatentModelModule(log_cond_pdf, encoder, decoder, sim_z, nchain, ctx)
        self.module.initialize(mx.init.Xavier(), ctx=ctx)
        self.module.hybridize()

        # Optimizer
        self.opt_params = self.module.collect_params()
        self.optimizer = None
        self.opt_loglik_params = self.module.loglik_params
        self.optimizer_dec_warmup = None
        self.optimizer_dec = None
        self.initial_lr_bc = 1e-3

        # Recorded loss function values
        # Log-likelihood part
        self.loss_ll = []
        # KL divergence part
        self.loss_kl = []
        # Bias-corrected loss function value
        self.loss_bc = []
        # Marginal loglikelihood
        self.nll = []

    # Reset recorded loss function values
    def init(self, lr=1e-3, lr_bc=1e-3):
        self.loss_ll = []
        self.loss_kl = []
        self.loss_bc = []
        self.nll = []
        self.initial_lr_bc = lr_bc

        self.optimizer = gluon.Trainer(
            self.opt_params, "adam", {"learning_rate": lr}
        )
        self.optimizer_dec_warmup = gluon.Trainer(
            self.opt_loglik_params, "adam", {"learning_rate": lr_bc}
        )
        self.optimizer_dec = gluon.Trainer(
            self.opt_loglik_params, "sgd", {"learning_rate": lr_bc, "momentum": 0.1, "wd": 0.001}
        )

    # Train the model
    def fit(self, x, epochs=100, batch_size=200, eval_nll=True, verbose=True):
        # Input data
        input = x.reshape((-1, self.module.input_dim))
        n = input.shape[0]
        x_ind = np.arange(n)
        total_time = 0.0

        for epoch in range(epochs):
            t1 = time.time()

            # Shuffle observations
            np.random.shuffle(x_ind)
            epoch_loss = 0.0

            # Update on mini-batches
            for i in range(0, n, batch_size):
                # Create mini-batch
                batch_id = x_ind[i:(i + batch_size)]
                bs = batch_id.size
                xsub_gpus = utils.split_and_load(input[batch_id, :], self.ctx)

                lossll_gpus = []
                losskl_gpus = []
                loss_gpus = []
                with autograd.record():
                    for xsub in xsub_gpus:
                        lossll, losskl = self.module(xsub)
                        lossll_gpus.append(lossll)
                        losskl_gpus.append(losskl)
                        loss_gpus.append(lossll + losskl)

                for l in loss_gpus:
                    l.backward()
                self.optimizer.step(bs)
                nd.waitall()

                lossll = np.sum([e.asscalar() for e in lossll_gpus])
                losskl = np.sum([e.asscalar() for e in losskl_gpus])
                epoch_loss += (lossll + losskl)
                self.loss_ll.append(lossll / bs)
                self.loss_kl.append(losskl / bs)

                # At the beginning of each epoch, print negative log-likelihood
                # values if eval_nll=True
                if i == 0 and eval_nll:
                    nll = self.module.negloglikelihood(input[batch_id, :], 50)
                    self.nll.append(nll)
                    if verbose:
                        print("===> epoch = {}, nll = {}\n".format(epoch, nll))

                if verbose:
                    print("epoch = {}, batch = {}, loss = {}".format(epoch, i // batch_size, (lossll + losskl) / bs))

            t2 = time.time()
            total_time += (t2 - t1)
            if verbose:
                print("\n===> epoch = {}, avg_loss = {}, time = {} s\n".format(epoch, epoch_loss / n, t2 - t1))

        if verbose:
            print("\nTraining took {} seconds with {} epochs".format(total_time, epochs))

    # Train the model using bias-corrected loss function
    def fit_bc(self, x, epochs=100, warmups=10, batch_size=200, burnin=100, step_size=0.1, pers_warm=True,
               eval_nll=True, verbose=True):
        # Input data
        input = x.reshape((-1, self.module.input_dim))
        n = input.shape[0]
        x_ind = np.arange(n)

        # Particles
        particles = nd.empty(shape=(self.module.nchain, n, self.module.latent_dim))

        total_time = 0.0
        for epoch in range(epochs):
            t1 = time.time()

            # Select the corresponding optimizer
            # We use the Adam optimizer for the warm-up stage
            if epoch < warmups:
                optimizer = self.optimizer_dec_warmup
            else:
                optimizer = self.optimizer_dec
                # Gradually decrease learning rate
                optimizer.set_learning_rate(self.initial_lr_bc / math.pow(epoch - warmups + 1.0, 0.5))

            # Also gradually decrease the Langevin step size
            langevin_ss = max(step_size / 10.0, step_size / math.pow(epoch + 1.0, 0.5))

            # Shuffle observations
            np.random.shuffle(x_ind)
            epoch_loss = 0.0

            # Update on mini-batches
            for i in range(0, n, batch_size):
                tt1 = time.time()

                # Create mini-batch
                batch_id = x_ind[i:(i + batch_size)]
                bs = batch_id.size
                if epoch == 0 or pers_warm is False:
                    init = None
                else:
                    init = particles[:, batch_id, :]

                loss_gpus, particles[:, batch_id, :] = self.module.bias_correction(input[batch_id, :], init=init,
                                                                                   burnin=burnin, step_size=langevin_ss)
                for l in loss_gpus:
                    l.backward()

                # Aggregate the gradient norm
                grad_norm = 0.0
                for name, param in self.opt_loglik_params.items():
                    if param.grad_req != "null":
                        for ctx in self.ctx:
                            grad_norm += param.grad(ctx=ctx).square().sum().asscalar()
                grad_norm = math.sqrt(grad_norm)

                # Record loss function value
                epoch_loss += grad_norm
                self.loss_bc.append(grad_norm / bs)
                optimizer.step(bs)

                tt2 = time.time()

                # At the beginning of each epoch, print negative log-likelihood
                # values if eval_nll=True
                if i == 0 and eval_nll:
                    nll = self.module.negloglikelihood(input[batch_id, :], 50)
                    self.nll.append(nll)
                    if verbose:
                        print("===> epoch = {}, lr = {}, nll = {}\n".format(epoch, optimizer.learning_rate, nll))

                if verbose:
                    print("epoch = {}, batch = {}, loss = {}, time = {} s".format(epoch, i // batch_size,
                                                                                  grad_norm / bs, tt2 - tt1))

            t2 = time.time()
            total_time += (t2 - t1)
            if verbose:
                print("\n===> epoch = {}, avg_loss = {}, time = {} s\n".format(epoch, epoch_loss / n, t2 - t1))

        if verbose:
            print("\nTraining took {} seconds with {} epochs".format(total_time, epochs))

        return particles

    # Simulate prior of theta
    # Output numpy array [nsim x p]
    def simulate_prior(self, nsim):
        return self.module.simulate_theta_prior(nsim)

    # Simulate posterior of theta
    # Input x [n x p]
    # Output numpy array size [n x nsim x p]
    def simulate_posterior(self, x, nsim):
        # Input data
        input = x.reshape((-1, self.module.input_dim))
        return self.module.simulate_theta_posterior(input, nsim)
