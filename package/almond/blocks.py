# Building blocks

from .utils import *
from mxnet import gluon
from mxnet.gluon import nn

# x => VAEEncoder => (mu, logvar) => z => VAEDecoder => u => ConditionalDistr => loglik
#                                         |                                 |
#                                         |<------- VAEDecoderLogLik ------>|

# Encoder part of the VAE model
# Input data => (mu, logvar)
class VAEEncoder(gluon.HybridBlock):
    # Constructor
    def __init__(self, enc_units=[1, 10], latent_dim=1, act="softrelu"):
        super(VAEEncoder, self).__init__()

        # Input dimension
        self.input_dim = enc_units[0]

        # Latent dimension (dimension of mu and logvar)
        self.latent_dim = latent_dim

        # From data to a common hidden state
        self.enc_common = MLPBuilder(enc_units, act)

        # From common hidden state to mu and logvar
        self.enc_mu = nn.Dense(latent_dim, in_units=self.enc_common.output_size())
        self.enc_logvar = nn.Dense(latent_dim, in_units=self.enc_common.output_size())

        # Activation function used
        if act == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.Activation(act)

    # Input: x [n x p]
    # Output: mean and log-variance of q(z|x)
    #         m(x) [n x d], log[v(x)] [n x d]
    #         d is the dimension of z
    def hybrid_forward(self, F, x):
        h = self.enc_common(x)
        h = self.act(h)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        # Safeguard: do not generate logvar that is too small or too large
        logvar = F.clip(logvar, -10.0, 10.0)
        return mu, logvar


# Decoder part of the VAE model
# Latent z => u=h(z)
class VAEDecoder(gluon.HybridBlock):
    # Constructor
    def __init__(self, dec_units=[10, 1], latent_dim=1, npar=1, act="softrelu"):
        super(VAEDecoder, self).__init__()

        # Output dimension
        self.output_dim = dec_units[-1]

        # Latent dimension
        self.latent_dim = latent_dim

        # How many parameters to output
        # Related to the conditional distribution p(x|theta)
        self.npar = npar

        # From latent z to theta
        # HybridSequential here is used as a list
        self.latent_to_dec = nn.HybridSequential()
        self.dec = nn.HybridSequential()
        for i in range(npar):
            self.latent_to_dec.add(nn.Dense(dec_units[0], in_units=latent_dim))
            self.dec.add(MLPBuilder(dec_units, act))

        # Activation function used
        if act == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.Activation(act)

    # Input: z ~ q(z|x) or z ~ p(z), [(nz x n) x d]
    # Output: u = h(z), [ [(nz x n) x p], ..., [(nz x n) x p] ]
    # Argument x => z
    def hybrid_forward(self, F, x):
        res = []
        for i in range(self.npar):
            h = self.latent_to_dec[i](x)
            h = self.act(h)
            fz = self.dec[i](h)
            res.append(fz)
        return res


# Conditional distribution f(x|u;theta)
# u, x => f(x|u;theta), theta is an additional parameter vector
class ConditionalDistr(gluon.HybridBlock):
    # Constructor
    def __init__(self):
        super(ConditionalDistr, self).__init__()

    # Input: x [N x p], u [ [N x d], ..., [N x d] ]
    # Output size [N]
    def hybrid_forward(self, F, x, u):
        raise NotImplementedError("subclasses must implement hybrid_forward()")


# VAEDecoder + ConditionalDistr
# Latent z => log[f(x|u;theta)], u=h(z)
class VAEDecoderLogLik(gluon.HybridBlock):
    # Constructor
    def __init__(self, decoder, nz, log_cond_pdf):
        super(VAEDecoderLogLik, self).__init__()

        # Decoder
        self.decoder = decoder

        # Number of latent variables simulated for each x
        self.nz = nz

        # log-pdf of X_i given u_i
        self.log_cond_pdf = log_cond_pdf

    # Input: z ~ q(z|x) or z ~ p(z), [(nz x n) x d]
    # Output: log[f(x|u;theta)], u=f(z), [1]
    # Argument x => z, xx => x
    def hybrid_forward(self, F, x, xx):
        u = self.decoder(x)
        # x [n x p], x_rep [(nz x n) x p]
        x_rep = rep_rows(xx, self.nz)
        loglik = F.sum(self.log_cond_pdf(x_rep, u))
        return loglik


# ===================== Langevin algorithm ===================== #

# Log-density of (x, z) used for Langevin sampling
# log[p(x, z)] = log[p(z)] + log[p(x|z)]
class LangevinLogPDF(gluon.HybridBlock):
    # Constructor
    def __init__(self, decoder, nchain, log_cond_pdf):
        super(LangevinLogPDF, self).__init__()

        # Decoder
        self.decoder = decoder

        # Number of chains in Langevin sampling
        self.nchain = nchain

        # log-pdf of X_i given u_i
        self.log_cond_pdf = log_cond_pdf

    # Input: z [nchain x n x d], x [n x p]
    # Output: log[p(x, z)]=log[p(z)]+log[p(x|z)], [1]
    # Argument x => z, xx => x
    def hybrid_forward(self, F, x, xx):
        z = x.reshape((-1, self.decoder.latent_dim))
        u = self.decoder(z)
        # x [n x p], x_rep [(nchain x n) x p]
        x_rep = rep_rows(xx, self.nchain)
        log_pxz = F.sum(self.log_cond_pdf(x_rep, u))
        log_pz = -0.5 * F.sum(F.square(z))
        log_pdf = log_pz + log_pxz
        return log_pdf


# Langevin sampling + log-likelihood loss function
# Langevin-sampled z => -log[p(x|theta)], theta=f(z)
class LangevinLoss(gluon.HybridBlock):
    # Constructor
    def __init__(self, decoder, nchain, log_cond_pdf):
        super(LangevinLoss, self).__init__()

        # Decoder
        self.decoder = decoder

        # Number of chains in Langevin sampling
        self.nchain = nchain

        # log-pdf of X_i given u_i
        self.log_cond_pdf = log_cond_pdf

    # Input: z [(nchain x n) x d]
    # Output: -log[f(x|u;theta)], u=f(z), [1]
    # Argument x => z, xx => x
    def hybrid_forward(self, F, x, xx):
        u = self.decoder(x)
        # x [n x p], x_rep [(nz x n) x p]
        x_rep = rep_rows(xx, self.nchain)
        loglik = F.sum(self.log_cond_pdf(x_rep, u))
        loss = -loglik / self.nchain
        return loss
