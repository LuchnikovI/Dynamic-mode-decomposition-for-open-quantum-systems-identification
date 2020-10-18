import tensorflow as tf
from utils import dmd
import math

class Embedding:
    '''This class provides tools for dmd based learning of
    markovian embaddings.'''
    def __init__(self):
        self.channel = None
        self.enc = None
        self.rank = None
        self.dec = None
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
        self.rank = lmbd.shape[0]
        self.dec = tf.reshape(right, (K, m**2, self.rank))[-1]
        self.channel = lmbd
        self.enc = tf.linalg.adjoint(left)

    def predict(self, history, total_time_steps):
        '''Simulates dynamics of a learned markovian embadding.
        Args:
            history: complex valued tensor of shape (K, m, m),
                history before prediction
            total_time steps: int value, number of time steps
        Returns:
            complex valued tensor of shape (total_time_steps, m, m)'''

        # will be filled by state per time step
        sys_states = []
        
        # initial state
        resh_history = tf.reshape(history, (-1,))
        state = tf.tensordot(self.enc, resh_history, axes=1)
        
        # simulation loop
        for i in range(total_time_steps):
            sys_state = tf.tensordot(self.dec, state, axes=1)
            sys_state = tf.reshape(sys_state, (self.sys_dim, self.sys_dim))
            sys_states.append(sys_state)
            state = self.channel * state
        sys_states = tf.convert_to_tensor(sys_states)
        return sys_states
