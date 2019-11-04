import numpy as np
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import nd

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
x = dat[0].reshape(-1, 784).asnumpy()
y = dat[1].asnumpy()
n = x.shape[0]

# PCA
pca = PCA(n_components=2)
pca.fit(x)

samp_ind = range(10000)
coords = pca.transform(x[samp_ind, :])
labels = y[samp_ind]

# Save coordinates and use R to plot
pdat = np.append(coords, labels.reshape(-1, 1), axis=1)
np.savetxt("../results/mnist-pca.txt", pdat)
