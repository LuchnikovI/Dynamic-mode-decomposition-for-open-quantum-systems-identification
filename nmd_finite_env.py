import tensorflow as tf
from utils import f_basis


class FiniteEnv:
    """Class provides tools for simulation open quantum dynamics with finite
    environment.
    Args:
        dim_sys: int value, dimension of system
        dim_mem: in value, dimension of environment"""

    def __init__(self, dim_sys, dim_mem):
        
        self.dim_sys = dim_sys
        self.dim_mem = dim_mem
        self.n = dim_sys * dim_mem
        self.gen = None

    def set_rand_gen(self,
                     dissipation_ampl,
                     hamiltonian_ampl):
        """Method generates random lindblad generator.
        Args:
            dissipation_ampl: float value, amplitude of dissipative part
            hamiltonian_ampl: float value, amplitude of hamiltonian part"""

        # identity matrix
        Id = tf.eye(self.n, dtype=tf.complex128)
        
        '''dissipator part'''
        # basis
        F = f_basis(self.n)
        # random spectrum of gamma matrix
        random_spec = tf.random.uniform((self.n**2-1,), dtype=tf.float64)
        random_spec = tf.cast(random_spec, dtype=tf.complex128)
        # random unitary
        U_re = tf.random.normal((self.n**2-1, self.n**2-1), dtype=tf.float64)
        U_im = tf.random.normal((self.n**2-1, self.n**2-1), dtype=tf.float64)
        U = tf.complex(U_re, U_im)
        U, _ = tf.linalg.qr(U)
        # gamma matrix
        gamma = (U * random_spec) @ tf.linalg.adjoint(U)
        # F * F^\dagger part of dissipator
        frhof = tf.einsum('qp,qij,pkl->ikjl',
                          gamma, F, tf.math.conj(F),
                          optimize='optimal')
        frhof = tf.reshape(frhof, (self.n ** 2, self.n ** 2))
        # antianti commutator part of dissipator
        FF = tf.einsum('qp,pki,qkj->ij', gamma, tf.math.conj(F), F)
        ffrho = tf.einsum('ij,kl->ikjl', FF, Id)
        rhoff = tf.einsum('ij,lk->ikjl', Id, FF)
        anti_com = 0.5 * (ffrho + rhoff)
        anti_com = tf.reshape(anti_com, (self.n ** 2, self.n ** 2))
        # dissipator
        diss = frhof - anti_com
        
        '''hamiltonian part'''
        # random hamiltonian
        H_re = tf.random.normal((self.n, self.n), dtype=tf.float64)
        H_im = tf.random.normal((self.n, self.n), dtype=tf.float64)
        H = tf.complex(H_re, H_im)
        H = 0.5 * (H + tf.linalg.adjoint(H))
        # comutator
        hrho = tf.einsum('ij,kl->ikjl', H, Id)
        rhoh = tf.einsum('ij,lk->ikjl', Id, H)
        com = 1j * (rhoh - hrho)
        com = tf.reshape(com, (self.n ** 2, self.n ** 2))
        
        '''total generator'''
        self.gen = hamiltonian_ampl * com + dissipation_ampl * diss

    def set_gen(self, gamma, H):
        """Method generates lindblad generator from given gamma matrix
        and Hamiltonian.
        Args:
            gamma: complex valued tensor of shape (n**2-1, n**2-1), matrix
                that defines dissipator
            H: complex valued tensor of shape (n, n), Hamiltonian"""
        
        # TODO: check whether it works or not
        # identity matrix
        Id = tf.eye(self.n, dtype=tf.complex128)
        
        '''dissipator part'''
        # basis
        F = f_basis(self.n)
        # F * F^\dagger part of dissipator
        frhof = tf.einsum('qp,qij,pkl->ikjl',
                          gamma, F, tf.math.conj(F),
                          optimize='optimal')
        frhof = tf.reshape(frhof, (self.n ** 2, self.n ** 2))
        # antianti commutator part of dissipator
        FF = tf.einsum('qp,pki,qkj->ij', gamma, tf.math.conj(F), F)
        ffrho = tf.einsum('ij,kl->ikjl', FF, Id)
        rhoff = tf.einsum('ij,lk->ikjl', Id, FF)
        anti_com = 0.5 * (ffrho + rhoff)
        anti_com = tf.reshape(anti_com, (self.n ** 2, self.n ** 2))
        # dissipator
        diss = frhof - anti_com
        
        '''hamiltonian part'''
        # comutator
        hrho = tf.einsum('ij,kl->ikjl', H, Id)
        rhoh = tf.einsum('ij,lk->ikjl', Id, H)
        com = 1j * (rhoh - hrho)
        com = tf.reshape(com, (self.n ** 2, self.n ** 2))
        
        '''total generator'''
        self.gen = com + diss

    def dynamics(self,
                 total_time,
                 time_step,
                 in_states):
        """Simulates the dynamics of a non-markovian system.
        Args:
            total_time: float value, total simulation time
            time_step: float value, simulation time step
            in_states: complex valued tensor of shape (N, sys_dim, sys_dim),
                where N is number of parralel experiments, sys_dim is the
                dimension of a system
        Returns:
            complex valued tensor of shape (N, time_steps, sys_dim, sys_dim),
            the dynamics of the system density matrix"""
        
        # TODO get rid of some tf.einsums
        # steady state of a lindbladian
        _, _, v = tf.linalg.svd(self.gen)
        steady_state = v[:, -1]
        steady_state = tf.reshape(steady_state, (self.n, self.n))
        steady_state = steady_state / tf.linalg.trace(steady_state)
        
        # steady state of a reservoir
        steady_state = tf.reshape(steady_state, (self.dim_sys,
                                                 self.dim_mem,
                                                 self.dim_sys,
                                                 self.dim_mem))
        steady_state = tf.einsum('kikj->ij', steady_state)
        
        # states of system + reservoir
        states = tf.einsum('qij,kl->qikjl', in_states, steady_state)
        states = tf.reshape(states, (-1, self.n**2))
        
        # quantum channel
        phi = tf.linalg.expm(time_step * self.gen)
        
        system_states = []  # list will be filled by dens. matrix vs time

        # simulation loop
        # TODO: tf.while_loop instead of standard for
        for _ in range(int(total_time / time_step)):
            
            system_state = tf.reshape(states, (-1, self.dim_sys,
                                               self.dim_mem,
                                               self.dim_sys,
                                               self.dim_mem))
            system_state = tf.einsum('qikjk->qij', system_state)
            system_states.append(system_state)
            states = tf.einsum('ij,qj->qi', phi, states)
            
        system_states = tf.convert_to_tensor(system_states)
        system_states = tf.transpose(system_states, (1, 0, 2, 3))
            
        return system_states
