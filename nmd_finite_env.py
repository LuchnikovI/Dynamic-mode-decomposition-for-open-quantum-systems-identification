import tensorflow as tf
from utils import f_basis

class FiniteEnv:
    
    def __init__(self, dim_sys, dim_mem):
        
        self.dim_sys = dim_sys
        self.dim_mem = dim_mem
        self.n = dim_sys * dim_mem
        self.gen = None
        
    def rand_gen(self,
                 dissipation_ampl,
                 hamiltonian_ampl):
       
        # identity matrix
        Id = tf.eye(self.n, dtype=tf.complex128)
        
        '''dissipator part'''
        # basis
        F = f_basis(self.n)
        # random spectrum of gamma matrix
        random_spec = tf.random.uniform((self.n,), dtype=tf.float64)
        random_spec = tf.cast(random_spec, dtype=tf.complex128)
        # random unitary
        U_re = tf.random.normal((self.n, self.n), dtype=tf.float64)
        U_im = tf.random.normal((self.n, self.n), dtype=tf.float64)
        U = tf.complex(U_re, U_im)
        U, _ = tf.linalg.qr(U)
        # gamma matrix
        gamma = (U * random_spec) * tf.linalg.adjoint(U)
        # F * F^\dagger part of dissipator
        frhof = tf.einsum('qp,qij,pkl->ikjl', gamma, F, tf.math.conj(F))
        frhof = tf.reshape(frhof, (self.n ** 2, self.n ** 2))
        # antianti commutator part of dissipator
        FF = tf.linalg.adjoint(F) @ F
        ffrho = tf.einsum('qp,qij,pkl->ikjl', gamma, FF, Id)
        rhoff = tf.einsum('qp,qij,plk->ikjl', gamma, Id, FF)
        anti_com = 0.5 * (ffrho + rhoff)
        anti_com = tf.reshape(anti_com, (self.n ** 2, self.n ** 2))
        # dissipator
        diss = frhof - 0.5 * (ffrho + rhoff)
        
        '''hamiltonian part'''
        # random hamiltonian
        H_re = tf.random.normal((self.n, self.n), dtype=tf.float64)
        H_im = tf.random.normal((self.n, self.n), dtype=tf.float64)
        H = tf.complex(H_re, H_im)
        H = 0.5 * (H - tf.linalg.adjoint(H))
        # comutator
        hrho = tf.einsum('ij,kl->ikjl', H, Id)
        rhoh = tf.einsum('ij,lk->ikjl', Id, H)
        com = 1j * (rhoh - hrho)
        com = tf.reshape(com, (self.n ** 2, self.n ** 2))
        
        '''total generator'''
        self.gen = hamiltonian_ampl * com + dissipation_ampl * diss
        