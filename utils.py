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


def hankel(T, K):
    """Return Hankel tensor from an ordinary tensor.
    Args:
        T: tensor of shape (batch_size, n, m)
        K: int value, depth of the Hankel matrix
    Returns:
        tensor of shape (batch_size, n-K+1, K, m)"""

    L = T.shape[1]
    i = tf.constant(1)
    t = T[:, tf.newaxis, :K]
    cond = lambda i, t: i<=L-K
    body = lambda i, t: [i+1, tf.concat([t, T[:, tf.newaxis, i:K+i]], axis=1)]
    _, t = tf.while_loop(cond, body, loop_vars=[i, t],
                  shape_invariants=[i.shape, tf.TensorShape([T.shape[0], None, K, T.shape[-1]])])
    return t
