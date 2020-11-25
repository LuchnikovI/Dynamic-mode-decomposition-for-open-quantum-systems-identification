import numpy as np
import scipy.linalg
import tensorflow as tf

sigma_x = np.matrix([[0,1],[1,0]])
sigma_y = np.matrix([[0,-1j],[1j,0]])
sigma_z = np.matrix([[1,0],[0,-1]])

def sample_spherical(npoints, ndim=3):
    '''Generates "npoints" uniformly distributed random points in 
    "ndim"-dimensioanl sphere
    '''
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def apply_unitary(rho, time, 
                  omega=1, theta=np.pi/2, phi=0):
    '''Applies single-qubit unitary rotation to input state "rho"
    on the angle "omega*time" around the axis determined by polar 
    angle "theta" and azimuthal angle "phi"
    '''
    sigma_n = (np.sin(theta) * np.cos(phi) * sigma_x +
               np.sin(theta) * np.sin(phi) * sigma_y +
               np.cos(theta) * sigma_z)
    evol_op = scipy.linalg.expm(-1j * sigma_n * omega * time / 2)
    return np.array(evol_op @ rho @ np.matrix(evol_op).H)

def apply_dephazing(rho, time, tau=1):
    '''Applies dephazing with characteristic time "tau"
    '''
    p = 1 - np.exp(- time/tau)
    return ((1 - p/2) * rho + p/2 * sigma_z @ rho @ sigma_z)

def apply_depolarizing(rho, time, tau=1):
    '''Applies depolarization with characteristic time "tau"
    '''
    p = 1 - np.exp(- time/tau)
    return (1 - p) * rho + p * np.eye(2) / 2 

def apply_damping(rho, time, tau=1):
    '''Applies damping with characteristic time "tau"
    '''
    p = 1 - np.exp(-time/tau)
    A1 = np.array([[1,0],[0,np.sqrt(1-p)]])
    A2 = np.array([[0,np.sqrt(p)],[0,0]])
    return A1 @ rho @ A1.transpose() + A2 @ rho @ A2.transpose()

def generate_dynamics(number_of_lines=1, 
                      total_time = 100,
                      time_step = 0.3,
                      omega=1, theta=np.pi/2, phi=0,
                      dec_type='deph',
                      tau=1,
                      mix_par=0.5):
    '''Generates dynamics which is a mixture of unitary evolution
    and decoherence process:
    evolution = P * unitary-evolution + (1-P) * dephazing
    Args:
        number_of_lines: number of random intial states processed
        total_time: total time of evolution
        time_step: time step for matrices generated
        omega: frequency of unitary evolution
        theta: polar angle of unitary qubit rotation
        phi: azimuthal angle of unitary qubit rotation
        tau: characterstic dephazing time
        mix_par: mixture paramter (P)
        dec_type: 'deph', 'depol', 'damp'
    Returns:
        numpy list of "number_of_lines" evolutions of random pure
        states'''
    lines = []
    for _ in range(number_of_lines):
        line_cur = []
        vec = sample_spherical(1)
        rho = np.eye(2)/2 + (sigma_x * vec[0][0] + 
                             sigma_y * vec[1][0] + 
                             sigma_z * vec[2][0])/2
        for time in np.arange(0, total_time, time_step):
            rho_cur1 = apply_unitary(rho, time, 
                                     omega=omega, theta=theta, phi=phi)
            if dec_type == 'deph':
                rho_cur2 = apply_dephazing(rho, time, tau=tau)
            elif dec_type == 'depol':
                rho_cur2 = apply_depolarizing(rho, time, tau=tau)
            elif dec_type == 'damp':
                rho_cur2 = apply_damping(rho, time, tau=tau)
            else:
                print('Decoherence type error')
            rho_cur = mix_par * rho_cur1 + (1 - mix_par) * rho_cur2
            line_cur.append(np.array(rho_cur))
        lines.append(line_cur)
    return tf.convert_to_tensor(lines)