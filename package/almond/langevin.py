import math
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd


# Prototype of a probility model
class ProbModel:
    def log_pdf(self, x, args):
        f = 0.0
        return f

    def log_pdf_grad(self, x, args):
        grad = nd.zeros_like(x)
        return grad


class Langevin:
    # Constructor
    def __init__(self, shape, ctx=mx.cpu()):
        # Device
        self.ctx = ctx
        # Shape of particles
        # In Langevin, the same as shape
        # In LangevinMultichain, add a "chain" dimension to shape
        self.shape = shape
        # Current position vector
        self.current = None

    # Generate a normal random vector
    def normal_noise(self, sd):
        return nd.random.normal(scale=sd, shape=self.shape, ctx=self.ctx)

    # Langevin diffusion
    def sample(self, model, start, step_size, num, burnin=10, args=None):
        # Initial position
        self.current = start
        # Result array
        res = nd.zeros(shape=(num - burnin, ) + self.shape, ctx=self.ctx)
        # Standard deviation of noise
        sd = math.sqrt(2.0 * step_size)
        # Diffusion
        for i in range(num):
            self.current = self.current + \
                           step_size * model.log_pdf_grad(self.current, args).detach() + \
                           self.normal_noise(sd)
            if i >= burnin:
                res[i - burnin] = self.current
        return res





# Prototype of a probility model
class ProbModelBatch:
    # If the parameter of interst has shape (d1, d2, ...), then
    # x is of shape (nbatch, d1, d2, ...)
    # Returns an array of length [nbatch]
    def log_pdf(self, x, args):
        f = nd.zeros(x.shape[0])
        return f

    def log_pdf_grad(self, x, args):
        grad = nd.zeros_like(x)
        return grad


class LangevinMultiChain(Langevin):
    # Constructor
    def __init__(self, shape, nchain, ctx=mx.cpu()):
        super(LangevinMultiChain, self).__init__(shape, ctx)

        # Number of independent chains
        self.nchain = nchain
        # Shape of particles
        self.shape = (nchain, ) + self.shape

    # Langevin diffusion
    def sample(self, model, start, step_size, burnin=10, args=None):
        # Check shape
        if start.shape != self.shape:
            raise ValueError("shape of 'start' is inconsistent with the sampler")
        # Initial position
        self.current = start
        # Standard deviation of noise
        sd = math.sqrt(2.0 * step_size)
        # Diffusion
        for i in range(burnin):
            self.current = self.current + \
                           step_size * model.log_pdf_grad(self.current, args).detach() + \
                           self.normal_noise(sd)
        return self.current





class Model1:
    def __init__(self, a, b, ctx=mx.cpu()):
        self.a = a
        self.b = b
        self.ctx = ctx

    def log_pdf(self, x, args):
        x1 = x[0].asscalar()
        x2 = x[1].asscalar()
        f = -(self.a - x1) ** 2 - self.b * (x2 - x1 * x1) ** 2
        return f

    def log_pdf_grad(self, x, args):
        x1 = x[0].asscalar()
        x2 = x[1].asscalar()
        dx1 = -2.0 * (x1 - self.a) - 4.0 * self.b * (x1 * x1 - x2) * x1
        dx2 = -2.0 * self.b * (x2 - x1 * x1)
        return nd.array([dx1, dx2], ctx=self.ctx)


# Multi-chain version
class Model2:
    def __init__(self, a, b, ctx=mx.cpu()):
        self.a = a
        self.b = b
        self.ctx = ctx

    # x: [nchain x 2]
    def log_pdf(self, x, args):
        x1 = x[:, 0]
        x2 = x[:, 1]
        f = -nd.square(self.a - x1) - self.b * nd.square(x2 - x1 * x1)
        return f

    def log_pdf_grad(self, x, args):
        x1 = x[:, 0]
        x2 = x[:, 1]
        dx1 = -2.0 * (x1 - self.a) - 4.0 * self.b * (x1 * x1 - x2) * x1
        dx2 = -2.0 * self.b * (x2 - x1 * x1)
        return nd.stack(dx1, dx2, axis=1)


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    np.random.seed(123)
    mx.random.seed(123)
    ctx = mx.cpu()

    model = Model1(0.3, 0.3, ctx=ctx)
    start = nd.array([0.0, 0.0], ctx=ctx)
    sampler = Langevin(start.shape, ctx=ctx)
    res = sampler.sample(model, start, step_size=0.1, num=1000, burnin=200)

    plt.scatter(res[:, 0].asnumpy(), res[:, 1].asnumpy())
    sns.jointplot(res[:, 0].asnumpy(), res[:, 1].asnumpy(), stat_func=None)

    np.random.seed(123)
    mx.random.seed(123)

    model = Model2(0.3, 0.3, ctx=ctx)
    nchain = 1000
    start = nd.random_normal(shape=(nchain, 2), ctx=ctx)
    sampler = LangevinMultiChain(start.shape[1:], nchain, ctx=ctx)
    res = sampler.sample(model, start, step_size=0.1, burnin=200)

    plt.scatter(res[:, 0].asnumpy(), res[:, 1].asnumpy())
    sns.jointplot(res[:, 0].asnumpy(), res[:, 1].asnumpy(), stat_func=None)
