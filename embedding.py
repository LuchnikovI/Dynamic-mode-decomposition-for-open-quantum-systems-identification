import tensorflow as tf
from utils import dmd
import math

class Embedding:
    '''This class provides tools for dmd based learning of
    markovian embadding.'''
    def __init__(self):
        self.channel = None
        self.enc = None
        self.min_rank = None
        self.sys_dim = None
        self.mem_dim = None
        self.K = None

    def learn(self, trajectories, K, eps=1e-5):
        '''Reconstructs markovian embedding from trajectories.
        Args:
            trajectories: complex valued tensor of shape (bs, n, m, m),
                where bs is number of trajectories, n is total number of
                time steps, m is size of density matrix
            K: int value, memory depth
            eps: tolerance'''

        # memory depth
        self.K = K
        # bs is number of trajectories
        # n is number of time steps
        # m is the size of a density matrix
        bs, n, m, _ = trajectories.shape
        self.sys_dim = m
        dtype = trajectories.dtype
        # dmd
        lmbd, right, left = dmd(trajectories, K, eps)
        lmbd = tf.cast(lmbd, dtype=dtype)
        r = lmbd.shape[0]
        self.min_rank = r
        # dimension of an effective reservoir
        eff_dim = int(math.sqrt(r)/m + 1)
        eff_r = (eff_dim ** 2) * (m ** 2)
        self.mem_dim = eff_dim
        dr = eff_r - r
        # adding extra dimension to elements of eigendecomposition
        lmbd = tf.concat([tf.zeros((dr,), dtype=dtype), lmbd],
                         axis=0)
        right = tf.concat([tf.zeros((right.shape[0], dr), dtype=dtype), right],
                          axis=1)
        left = tf.concat([tf.zeros((left.shape[0], dr), dtype=dtype), left],
                         axis=1)
        dec = tf.reshape(right, (K, m**2, eff_r))[-1]
        # trace operator
        ptrace = tf.eye(eff_dim * m ** 2, dtype=dtype)
        ptrace = tf.reshape(ptrace, (m, m, eff_dim, m, m, eff_dim))
        ptrace = tf.transpose(ptrace, (0, 1, 3, 2, 4, 5))
        ptrace = tf.reshape(ptrace, (m**2, eff_r))
        # attempt to build quantum channel from the spectrum of T
        s, u, v = tf.linalg.svd(ptrace)
        s = tf.cast(s, dtype=dtype)
        s_inv = 1 / s
        Q = (v * s_inv) @ tf.linalg.adjoint(u) @ dec + tf.eye(eff_r, dtype=dtype) -\
        v @ tf.linalg.adjoint(v)
        self.channel = (Q * lmbd) @ tf.linalg.inv(Q)
        self.enc = Q @ tf.linalg.adjoint(left)

    def predict(self, history, total_time_steps, ind, u):
        '''Simulates dynamics of a learned markovian embadding.
        Args:
            history: complex valued tensor of shape (K, m, m),
                history before prediction
            total_time steps: int value, number of time steps
            ind: int number, discrete time moment when to apply control
                signal
            u: complex valued tensor of shape (m, m), unitary matrix
                representing control
        Returns:
            complex valued tensor of shape (total_time_steps, m, m)'''

        # will be filled by state per time step
        sys_states = []
        
        # initial state
        resh_history = tf.reshape(history, (-1,))
        state = tf.tensordot(self.enc, resh_history, axes=1)
        
        # simulation loop
        for i in range(total_time_steps):
            sys_state = tf.reshape(state, (self.sys_dim,
                                           self.mem_dim,
                                           self.sys_dim,
                                           self.mem_dim))
            sys_state = tf.einsum('ikjk->ij', sys_state)
            sys_states.append(sys_state)
            state = tf.tensordot(self.channel, state, axes=1)
            if i == ind:
                U = tf.tensordot(u, tf.eye(self.mem_dim, dtype=u.dtype), axes=0)
                U = tf.transpose(U, (0, 2, 1, 3))
                U = tf.reshape(U, (self.sys_dim*self.mem_dim,
                                   self.sys_dim*self.mem_dim))
                U = tf.tensordot(U, tf.math.conj(U), axes=0)
                U = tf.transpose(U, (0, 2, 1, 3))
                U = tf.reshape(U, ((self.sys_dim*self.mem_dim)**2,
                                   (self.sys_dim*self.mem_dim)**2))
                state = tf.tensordot(U, state, axes=1)
        sys_states = tf.convert_to_tensor(sys_states)
        return sys_states
