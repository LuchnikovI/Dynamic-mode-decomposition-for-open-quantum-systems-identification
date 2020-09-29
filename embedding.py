import tensorflow as tf
from utils import hankel
from dmd import dmd
import math

class Embedding:

    def __init__(self):
        self.channel = None
        self.enc = None

    def learn(self, trajectories, K):
        # bs is batch size
        # n is number of time steps
        # m is the size of density matrix
        bs, n, m, _ = trajectories.shape
        dtype = trajectories.dtype
        # reshape density matrices to vectors
        t = tf.reshape(trajectories, (bs, n, m**2))
        # build hankel matrix of shape (bs, n-K+1, K, m**2)
        t = hankel(t, K)
        # build X and Y matrices, both have shape (K*(m**2), bs, n-K)
        t = tf.reshape(t, (bs, n-K+1, K*(m**2)))
        t = tf.transpose(t, (2, 0, 1))
        X = t[..., :-1]
        Y = t[..., 1:]
        # calculate T matrix in the form of spectral decomposition
        # lmbd is the spectrun that has shape (r,)
        # right and left are right and left eigenvectors, both have shape
        # (K*(m**2), r)
        lmbd, right, left = dmd(X, Y)
        lmbd = tf.cast(lmbd, dtype=dtype)
        r = lmbd.shape[0]
        # dimension of an effective reservoir
        eff_dim = int(math.sqrt(r)/m)
        dec = tf.reshape(right, (K, m**2, r))[-1]
        # trace operator
        '''ptrace = tf.eye(eff_dim * m ** 2, dtype=dtype)
        ptrace = tf.reshape(ptrace, (m, m, eff_dim, m, m, eff_dim))
        ptrace = tf.transpose(ptrace, (0, 1, 3, 2, 4, 5))
        ptrace = tf.reshape(ptrace, (m**2, r))
        s, u, v = tf.linalg.svd(ptrace)
        s = tf.cast(s, dtype=dtype)
        s_inv = 1 / s
        Q = (v * s_inv) @ tf.linalg.adjoint(u) @ dec + tf.eye(r, dtype=dtype) -\
        v @ tf.linalg.adjoint(v)
        self.channel = (Q * lmbd) @ tf.linalg.inv(Q)
        self.enc = tf.linalg.adjoint(left)'''
        return r