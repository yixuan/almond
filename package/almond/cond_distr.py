# Common conditional distributions

import math
from .blocks import ConditionalDistr

# log[f(x_i|theta_i)], X_i|u_i ~ N(mu_i, 1), u = [mu]
# Input: x [N x p], u [N x p]
# Output size [N]
class ConditionalNormal(ConditionalDistr):
    # Constructor
    def __init__(self, dimu, sigma=1.0):
        super(ConditionalNormal, self).__init__()
        self.sigma2 = sigma * sigma
        self.const = 0.5 * dimu * math.log(2.0 * math.pi * self.sigma2)

    # Input: x [N x p], u [ [N x p] ]
    # Output size [N]
    def hybrid_forward(self, F, x, u):
        mu = u[0]
        log_pdf = -(0.5 / self.sigma2) * F.sum(F.square(mu - x), axis=1) - self.const
        return log_pdf


# log[f(x_i|theta_i)], X_i|u_i ~ N(mu_i, var_i), u = [mu, logvar]
# Input: x [N x p], u [N x p, N x p]
# Output size [N]
class ConditionalNormalTwoPar(ConditionalDistr):
    # Constructor
    def __init__(self, dimu):
        super(ConditionalNormalTwoPar, self).__init__()
        self.const = 0.5 * dimu * math.log(2.0 * math.pi)

    # Input: x [N x p], u [ [N x p], [N x p] ]
    # Output size [N]
    def hybrid_forward(self, F, x, u):
        mu = u[0]
        logvar = u[1]
        var = logvar.exp() + 0.001
        logvar = var.log()

        log_pdf = -0.5 * F.sum(F.square(x - mu) / var + logvar, axis=1) - self.const
        return log_pdf


# log[f(x_i|u_i)], X_i|u_i ~ Bernoulli(sigmoid(u_i)), u = [u]
# Input: x [N x p], u [N x p]
# Output size [N]
class ConditionalBernoulli(ConditionalDistr):
    # Constructor
    def __init__(self):
        super(ConditionalBernoulli, self).__init__()

    # Input: x [N x p], u [ [N x p] ]
    # Output size [N]
    def hybrid_forward(self, F, x, u):
        # log[p(x|u)] = x * log(gt) + (1 - x) * log(1 - gt)
        # gt = sigmoid(u) => log(gt) = logsigmoid(u)
        # 1 - gt = exp(-u) * sigmoid(u) => log(1 - gt) = -u + logsigmoid(u)
        # log[f(x|u)] = x * logsigmoid(u) + (1 - x) * (-u + logsigmoid(u))
        #             = -u + logsigmoid(u) + x * u
        #             = (x - 1) * u + logsigmoid(u)
        u = F.clip(u[0], -20.0, 20.0)
        log_pdf = F.sum((x - 1.0) * u + F.log(F.sigmoid(u)), axis=1)
        return log_pdf
