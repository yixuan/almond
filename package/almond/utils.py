# Utility functions

import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn

# Given a tensor x, compute log-sum-exp along the specified axis
# log(exp(x1) + ... + exp(xn))
# = c + log(exp(x1 - c) + ... + exp(xn - c)), c = max(x)
def log_sum_exp(x, axis, keepdims=False, F=nd):
    # Compute c on the corresponding axis
    c = F.max(x, axis=axis, keepdims=True)
    # Subtract ci from the original tensor
    xc = F.broadcast_sub(x, c)
    # Compute exp and sum, and add c back
    res = F.broadcast_add(F.log(F.sum(xc.exp(), axis=axis, keepdims=True)), c)
    if not keepdims:
        res = F.squeeze(res, axis=axis)
    return res


# Repeat rows of a 2-D tensor
def rep_rows(x, nrep):
    # If nrep == 1, return x itself to avoid copying
    if nrep == 1:
        return x
    return x.tile((nrep, 1))


# Generate an MLP given the network structure
class MLPBuilder(gluon.HybridBlock):
    # Constructor
    def __init__(self, units=[1, 20], act="softrelu"):
        super(MLPBuilder, self).__init__()

        self.in_size = units[0]
        self.out_size = units[-1]
        self.net = nn.HybridSequential()
        if act == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.Activation(act)

        # Add linear layers
        nlayer = len(units)
        with self.net.name_scope():
            for i in range(1, nlayer - 1):
                self.net.add(nn.Dense(units[i], in_units=units[i - 1]))
                self.net.add(self.act)
            # Do not apply activation function in the last layer
            self.net.add(nn.Dense(units[-1], in_units=units[-2]))

    # Apply the MLP to a data set
    def hybrid_forward(self, F, x):
        return self.net(x)

    # Size of flattened input
    def input_size(self):
        return self.in_size

    # Size of flattened output
    def output_size(self):
        return self.out_size
