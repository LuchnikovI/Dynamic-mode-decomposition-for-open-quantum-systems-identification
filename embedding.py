import tensorflow as tf
from utils import dmd
from utils import pinv


class Embedding:
    '''This class provides tools for dmd based learning of
    markovian embadding.'''
    def __init__(self):
        self.channel = None
        self.enc = None
        self.dec = None
        self.dec_pinv = None
        self.res_dec = None
        self.rank = None
        self.sys_dim = None
        self.rank = None
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
        self.rank = r
        self.dec = tf.reshape(right, (K, m**2, r))[-1]
        self.channel = lmbd
        self.enc = tf.linalg.adjoint(left)
        self.dec_pinv = pinv(self.dec)
        self.res_dec = tf.eye(self.dec.shape[1], dtype=dtype) - self.dec_pinv @ self.dec

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
            sys_state = tf.tensordot(self.dec, state)
            sys_state = tf.reshape(state, (self.sys_dim,
                                           self.sys_dim))
            sys_states.append(sys_state)
            state = tf.tensordot(self.channel, state, axes=1)
            if i == ind:
                U = tf.tensordot(u, tf.math.conj(u), axes=0)
                U = tf.transpose(U, (0, 2, 1, 3))
                U = tf.reshape(U, (self.sys_dim**2,
                                   self.sys_dim**2))
                U = self.res_dec + self.dec_pinv @ U @ self.dec
                state = tf.tensordot(U, state, axes=1)
        sys_states = tf.convert_to_tensor(sys_states)
        return sys_states
