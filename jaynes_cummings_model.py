import tensorflow as tf
import numpy as np
from tqdm import tqdm

class JC_model:
    """ Class provides the tool for Jaynes-Cummings model simulation.
    Args:
    - sys_dim, system dimention, type: int
    - mode_type, defines environment state,
      type: string, possible settings:
            - Fock state
            - Coherent state
            - Thermal state
    - mode_par, parameter of the environment state,
      type: - 'int', for Fock state
            - 'complex', for Coherent state
            - 'float', for Thermal state
    """

    def __init__(self, sys_dim, mode_type, mode_par):
        self.sys_dim = sys_dim
        self.mode_type = mode_type
        self.mode_par = mode_par
        self.env_dim = self.minimal_env_dim()
        self.env_state = self.env_init()

        sigma_x = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)
        sigma_y = tf.constant([[0, -1j], [1j, 0]], dtype=tf.complex128)
        sigma_z = tf.constant([[1, 0], [0, -1]], dtype=tf.complex128)

        self.pauli = tf.concat([sigma_x[tf.newaxis],
                                sigma_y[tf.newaxis],
                                sigma_z[tf.newaxis]], axis=0)
        self.generator = None


    def minimal_env_dim(self):
        """ Return the minimal dimetion of the environmet
            with respect to the field type. """
        if self.mode_type == 'Fock_state':
            return  self.mode_par + 2
        elif self.mode_type == 'Coherent_state':
            return int(3 * abs(self.mode_par) ** 2) + 2
        elif self.mode_type == 'Thermal_state':
            mean = 1/(np.exp(self.mode_par) - 1)
            return int(mean + 3 * mean / (1 + mean)) + 2
        else:
            print('Unknown mode type')


    def field_mode_operators(self):
        """ Returns field mode creation (a†) and annihilation (a) operators. """
        arr = tf.math.sqrt(tf.range(
            0., self.env_dim, dtype=tf.float64))
        matr = tf.reshape(tf.concat(
            [arr for i in range(self.env_dim)], 0), [self.env_dim, self.env_dim])
        annih_oper = tf.cast(tf.linalg.band_part(
            matr, 0, 1) - tf.linalg.band_part(
            matr, 0, 0), dtype=tf.complex128)
        return annih_oper, tf.linalg.adjoint(annih_oper)


    def env_init(self):
        """ Return the initial state of environment. """
        if self.mode_type == 'Fock_state':
            fock_density = tf.einsum(
                'i,j->ij', tf.one_hot(
                    int(self.mode_par), self.env_dim), tf.math.conj(
                    tf.one_hot(int(self.mode_par), self.env_dim)))
            return tf.cast(fock_density, dtype=tf.complex128)

        elif self.mode_type == 'Coherent_state':
            annihilation, creation = self.field_mode_operators()
            displacement = tf.linalg.expm(
                self.mode_par * creation - np.conj(self.mode_par) * annihilation)
            alpha_state = tf.einsum(
                'ij, j->i', displacement, tf.cast(
                    tf.one_hot(0, self.env_dim), dtype=tf.complex128))
            coherent_density = tf.einsum(
                'i,j->ij', alpha_state, tf.math.conj(alpha_state))
            return tf.cast(coherent_density, dtype=tf.complex128)

        elif self.mode_type == 'Thermal_state':
            thermal_density = tf.reduce_sum(
                [tf.math.exp(- self.mode_par * float(n)) * tf.einsum(
                    'i,j->ij', tf.one_hot(n, self.env_dim), tf.math.conj(
            tf.one_hot(n, self.env_dim))) for n in range(self.env_dim)], axis=0)
            return tf.cast(
                thermal_density / tf.linalg.trace(
                    thermal_density), dtype=tf.complex128)

        else:
            print('Unknown mode type')


    def lindblad_generator(self, alpha, omega, gamma):
        """ Return Lindblad generator of Jaynes-Cummings model
            with dissipation.
            Args:
            - alpha, system Hamiltonian decomposition
            - omega, field oscillation frequency
            - gamma, damping amplitude """

        # TODO simulation for qudit system

        dim = self.sys_dim * self.env_dim
        id_env = tf.eye(self.env_dim, self.env_dim, dtype=tf.complex128)
        id_sys = tf.eye(self.sys_dim, self.sys_dim, dtype=tf.complex128)
        identity = tf.eye(dim, dim, dtype=tf.complex128)

        # Creation & Annihilation operators
        annihilation, creation = self.field_mode_operators()

        # System Hamiltonian
        h_sys = tf.reshape(
            tf.einsum('ij,kl->ikjl', tf.reduce_sum(
        [alpha[i] * self.pauli[i] for i in range(3)], axis=0), id_env), (dim, dim))

        # Field Hamiltonian
        h_field = tf.reshape(tf.einsum(
            'ij,kl->ikjl', id_sys, creation @ annihilation), (dim, dim))

        # Interaction Hamiltonian
        h_int = omega * tf.reshape(
            tf.einsum('ij,kl->ikjl', 1/2 * (
            self.pauli[0] - 1j * self.pauli[1]), annihilation) +\
            tf.einsum('ij,kl->ikjl', 1/2 * (
            self.pauli[0] + 1j * self.pauli[1]), creation), (dim, dim))

        jc_ham = h_sys + h_field + h_int

        # Commutator vectorization
        ham_id = tf.einsum('ij,kl->ikjl', jc_ham, identity)
        id_ham = tf.einsum('ij,lk->ikjl', identity, jc_ham)
        commutator = tf.reshape(
            1j * (id_ham - ham_id), (dim ** 2, dim ** 2))

        # Jump operators in vectorized form
        annihilation_jump = tf.reshape(tf.einsum(
            'ij,kl->ikjl', id_sys, annihilation), (dim, dim))
        creation_jump = tf.reshape(tf.einsum(
            'ij,kl->ikjl', id_sys, creation), (dim, dim))

        # Dissipator in vectorized form 
        dissipator = gamma * (
            tf.reshape(tf.einsum('ij,kl->ikjl', tf.transpose(
                creation_jump), annihilation_jump), (
                    dim ** 2, dim ** 2)) -\
            1/2 * tf.reshape(tf.einsum(
                'ij,kl->ikjl', identity, creation_jump @ annihilation_jump), (
                    dim ** 2, dim ** 2)) -\
            1/2 * tf.reshape(tf.einsum(
                'ij,kl->ikjl', tf.transpose(
                    creation_jump @ annihilation_jump), identity), (
                        dim ** 2, dim ** 2))
        )
        self.generator = commutator + dissipator


    def sample_spherical(self, npoints, ndim=3):
        """ Generates "npoints" uniformly distributed random points in
            "ndim"-dimensioanl sphere """
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec


    def generate_dynamics(self, number_of_lines,
                          total_time, time_step, rho_0=None):
        """ Generate dynamics
            Args:
            - number_of_lines, number of different trajectories
            - total_time, total simulation time
            - time_step, simulation step
            - rho_0, initial state of the system,
                     if not given the system will
                     be initialized randomly """
        dim = self.sys_dim * self.env_dim
        if rho_0 == None:
            lines = []
            for _ in range(number_of_lines):
                line = []
                vec = self.sample_spherical(1)
                sys_state = 1/2 * (tf.cast(
                    tf.eye(2), dtype=tf.complex128) + tf.reduce_sum(
                    [vec[i] * self.pauli[i] for i in range(3)], axis=0))

                initial_state = tf.reshape(tf.einsum(
                    'ij,kl->ikjl', sys_state, self.env_state), (dim, dim))

                for time in tqdm(np.arange(0., total_time, time_step)):
                    state = tf.linalg.expm(
                        time * self.generator) @ tf.reshape(
                            initial_state, (dim ** 2, 1))
                    line.append(tf.einsum('ikjk->ij', tf.reshape(
                        state, (self.sys_dim, self.env_dim, self.sys_dim, self.env_dim))))
                lines.append(line)
            self.dynamics = tf.convert_to_tensor(lines)

        else:
            line = []
            initial_state = tf.reshape(tf.einsum(
                'ij,kl->ikjl', rho_0, self.env_state), (dim, dim))
            for time in tqdm(np.arange(0., total_time, time_step)):
                state = tf.linalg.expm(
                    time * self.generator) @ tf.reshape(
                        initial_state,  (dim ** 2, 1))
                line.append(tf.einsum('ikjk->ij', tf.reshape(
                    state, (self.sys_dim, self.env_dim, self.sys_dim, self.env_dim))))
            self.dynamics = tf.convert_to_tensor(line)

