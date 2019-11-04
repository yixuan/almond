import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
import matplotlib.pyplot as plt

from almond import LatentModel, ConditionalBernoulli

# Number of GPUs for computation
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
# Otherwise, use CPU
# ctx = [mx.cpu()]


# Input data => (mu, logvar)
class MNISTEncoder(gluon.HybridBlock):
    # Constructor
    def __init__(self, nfilter=32, latent_dim=10, act="elu"):
        super(MNISTEncoder, self).__init__()

        self.conv_input_shape = (1, 28, 28)
        self.input_dim = 28 * 28
        self.latent_dim = latent_dim
        if act == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.Activation(act)

        # [28 x 28] => [14 x 14]
        self.conv1 = nn.Conv2D(nfilter, in_channels=1, kernel_size=4, strides=2, padding=1)
        self.bn1 = nn.BatchNorm(in_channels=nfilter)
        # [14 x 14] => [7 x 7]
        self.conv2 = nn.Conv2D(nfilter, in_channels=nfilter, kernel_size=4, strides=2, padding=1)
        self.bn2 = nn.BatchNorm(in_channels=nfilter)
        # [7 x 7] => [3 x 3]
        self.conv3 = nn.Conv2D(nfilter, in_channels=nfilter, kernel_size=4, strides=2, padding=1)
        self.bn3 = nn.BatchNorm(in_channels=nfilter)
        # [3 x 3] => 1024
        self.fc = nn.Dense(1024, in_units=nfilter * 3 * 3)
        # To mu and logvar
        self.enc_mu = nn.Dense(latent_dim, in_units=1024)
        self.enc_logvar = nn.Dense(latent_dim, in_units=1024)

    # Does not apply activation function to the output
    def hybrid_forward(self, F, x):
        h1 = self.bn1(self.conv1(x.reshape((-1, ) + self.conv_input_shape)))
        h1 = self.act(h1)

        h2 = self.bn2(self.conv2(h1))
        h2 = self.act(h2)

        h3 = self.bn3(self.conv3(h2))
        h3 = self.act(h3)

        h4 = self.fc(h3)
        h4 = self.act(h4)

        mu = self.enc_mu(h4)
        logvar = self.enc_logvar(h4)
        return mu, logvar


# Latent z => theta=f(z)
class MNISTDecoderOnePar(gluon.HybridBlock):
    # Constructor
    def __init__(self, nfilter=32, latent_dim=10, act="elu"):
        super(MNISTDecoderOnePar, self).__init__()

        self.conv_input_shape = (nfilter, 3, 3)
        self.output_dim = 28 * 28
        self.latent_dim = latent_dim
        if act == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.Activation(act)

        # latent_dim => 1024
        self.latent_to_fc = nn.Dense(1024, in_units=latent_dim)
        # 1024 => [3 x 3]
        self.fc_to_dec = nn.Dense(nfilter * 3 * 3, in_units=1024)
        # [3 x 3] => [7 x 7]
        self.conv1 = nn.Conv2DTranspose(nfilter, in_channels=nfilter, kernel_size=3, strides=2, padding=(0, 0))
        self.bn1 = nn.BatchNorm(in_channels=nfilter)
        # [7 x 7] => [14 x 14]
        self.conv2 = nn.Conv2DTranspose(nfilter, in_channels=nfilter, kernel_size=4, strides=2, padding=(1, 1))
        self.bn2 = nn.BatchNorm(in_channels=nfilter)
        # [14 x 14] => [28 x 28]
        self.conv3 = nn.Conv2DTranspose(nfilter, in_channels=nfilter, kernel_size=4, strides=2, padding=(1, 1))
        self.bn3 = nn.BatchNorm(in_channels=nfilter)
        # [28 x 28] => [28 x 28]
        self.conv4 = nn.Conv2D(1, in_channels=nfilter, kernel_size=3, padding=1)

    # Does not apply activation function to the output
    def hybrid_forward(self, F, x):
        h = self.latent_to_fc(x)
        h = self.act(h)

        h = self.fc_to_dec(h)
        h = self.act(h)

        h1 = self.bn1(self.conv1(h.reshape((-1, ) + self.conv_input_shape)))
        h1 = self.act(h1)

        h2 = self.bn2(self.conv2(h1))
        h2 = self.act(h2)

        h3 = self.bn3(self.conv3(h2))
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        return h4.reshape((-1, self.output_dim))


class MNISTDecoder(gluon.HybridBlock):
    def __init__(self, nfilter=32, latent_dim=10, npar=2, act="elu"):
        super(MNISTDecoder, self).__init__()

        self.output_dim = 28 * 28
        self.latent_dim = latent_dim
        self.npar = npar
        self.decs = nn.HybridSequential()
        for i in range(npar):
            self.decs.add(MNISTDecoderOnePar(nfilter, latent_dim, act))

    def hybrid_forward(self, F, x):
        return [self.decs[i].hybrid_forward(F, x) for i in range(self.npar)]


# Data
path = "./data"
np.random.seed(123)
mx.random.seed(123)

def transform(data, label):
    data = nd.moveaxis(data, 2, 0).astype("float32") / 255
    label = label
    return data, label

train = mx.gluon.data.vision.datasets.MNIST(path, train=True, transform=transform)
# A large batch size to retrieve all data
loader = mx.gluon.data.DataLoader(train, batch_size=1e5, shuffle=True)

dat = list(loader)[0]
x = dat[0]
y = dat[1]
n = x.shape[0]

# Plot 100 digits
def vis100(dat, filename):
    dat = dat.reshape(-1, 784)
    samp = np.clip(dat[0:100, :].asnumpy(), 0.0, 1.0).reshape(100, 28, 28)
    pic = samp.reshape(10, 280, 28).transpose(1, 0, 2).reshape(280, 280)

    fig = plt.figure(figsize=(8, 8))
    sub = fig.add_subplot(111)
    sub.axes.get_xaxis().set_visible(False)
    sub.axes.get_yaxis().set_visible(False)
    sub.imshow(pic, cmap="gray")
    # fig.show()
    plt.savefig(filename, bbox_inches="tight")

vis100(x, "../results/mnist-train-100.pdf")

# Model
np.random.seed(123)
mx.random.seed(123)
nfilter = 64
latent_dim = 20
model = LatentModel(ConditionalBernoulli(),
                    encoder=MNISTEncoder(nfilter=nfilter, latent_dim=latent_dim, act="softrelu"),
                    decoder=MNISTDecoder(nfilter=nfilter, latent_dim=latent_dim, npar=1, act="softrelu"),
                    sim_z=10, nchain=20, ctx=ctx)
model.init(lr=0.0001, lr_bc=0.0001)

# Training
np.random.seed(123)
mx.random.seed(123)
epochs = 500
batch_size = 200
filename = "../results/mnist-{}-{}-ep{}.mx".format(nfilter, latent_dim, epochs)

# VAE
model.fit(x, epochs=epochs, batch_size=batch_size, eval_nll=False, verbose=True)
model.module.save_parameters(filename)
# model.module.load_parameters(filename)

# Generate random images
def vis100(model, filename):
    z = mx.nd.random.normal(shape=(100, model.module.latent_dim), ctx=ctx[0])
    fz = model.module.dec(z)[0]
    n = fz.shape[0]
    mu = nd.sigmoid(fz.reshape((n, -1))).asnumpy()
    samp = np.clip(mu, 0.0, 1.0).reshape(100, 28, 28)
    pic = samp.reshape(10, 280, 28).transpose(1, 0, 2).reshape(280, 280)

    fig = plt.figure(figsize=(8, 8))
    sub = fig.add_subplot(111)
    sub.axes.get_xaxis().set_visible(False)
    sub.axes.get_yaxis().set_visible(False)
    sub.imshow(pic, cmap="gray")
    # fig.show()
    plt.savefig(filename, bbox_inches="tight")
    return fig, samp

np.random.seed(123)
mx.random.seed(123)
filename = "../results/mnist-{}-{}-ep{}.pdf".format(nfilter, latent_dim, epochs)
fig, samp = vis100(model, filename)

# ALMOND bias correction
np.random.seed(123)
mx.random.seed(123)
epochs_bc = 200
batch_size = 200
burnin = 5
ss = 0.02
filename = "../results/mnist-{}-{}-ep{}-bc{}-burnin{}-ss{}.mx".format(nfilter, latent_dim, epochs, epochs_bc, burnin, ss)

particles = model.fit_bc(x, epochs=epochs_bc, warmups=epochs_bc, batch_size=batch_size,
                         burnin=burnin, step_size=ss,
                         eval_nll=False, verbose=True)
model.module.save_parameters(filename)
# model.module.load_parameters(filename)

np.random.seed(123)
mx.random.seed(123)
filename = "../results/mnist-{}-{}-ep{}-bc{}-burnin{}-ss{}.pdf".format(nfilter, latent_dim, epochs, epochs_bc, burnin, ss)
fig, samp = vis100(model, filename)
