import tensorflow as tf
import math


def f_basis(n, dtype=tf.complex128):
    """The function returns basis in the space of real traceless matrices
    of size n. For all matrices, the following condition holds true 
    <F_i, F_j> = I_ij, where I is the identity matrix.
    Args:
        n: int value, dimension of a space
        dtype: type of matrices
    Returns:
        tensor of shape (n**2-1, n, n), 0th index enumerates matrices"""

    F = tf.eye(n ** 2, dtype=dtype)
    F = tf.reshape(F, (n ** 2, n, n))[:-1]
    F = tf.reshape(F, (n-1, n+1, n, n))[:, 1:]
    F00 = tf.ones((n, 1), dtype=dtype) / math.sqrt(n)
    diag = tf.concat([F00, tf.eye(n, n-1, dtype=dtype)], axis=1)
    q, _ = tf.linalg.qr(diag)
    diag = tf.linalg.diag(tf.transpose(q))[1:]
    diag = diag[:, tf.newaxis]
    F = tf.concat([diag, F], axis=1)
    F = tf.reshape(F, (-1, n, n))
    return F