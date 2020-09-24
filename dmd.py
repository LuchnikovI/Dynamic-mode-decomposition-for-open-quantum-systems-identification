import tensorflow as tf


def dmd(X, Y, eps=1e-5):
    """Solves the following linear regression problem
    ||TX - Y||_F --> min with respect to transition matrix T.
    Matrix T is found by using dynamic mode decomposition (dmd) in the form
    of its eigendecomposition with the minimal possible rank.
    You may read more about dmd in the following paper
    https://arxiv.org/pdf/1312.0041.pdf
    Args:
        X: tensor of shape(n, ...)
        Y: tensor of shape(n, ...)
        eps: float value, tolerance that defines rank
    Returns:
        three tensors of shapes (r,), (n, r), and (n, r),
        dominant eigenvalues and corresponding (right and left)
        eigenvectors
    Note:
        n -- dimension of one data point, r -- rank that is determined
        by tolerance eps."""

    dtype = X.dtype
    X_resh = tf.reshape(X, (X.shape[-1], -1))
    Y_resh = tf.reshape(Y, (Y.shape[-1], -1))
    # SVD of X_resh matrix
    lmbd, u, v = tf.linalg.svd(X_resh)
    # number of singular vals > eps
    ind = tf.reduce_sum(tf.cast(lmbd > eps, dtype=tf.int32))
    # truncation of all elements of the svd
    lmbd = lmbd[:ind]
    lmbd_inv = 1 / lmbd
    lmbd_inv = tf.cast(lmbd_inv, dtype=dtype)
    u = u[:, :ind]
    v = v[:, :ind]
    # eigendecomposition of T_tilda
    T_tilda = tf.linalg.adjoint(u) @ Y_resh @ (v * lmbd_inv)
    eig_vals, right = tf.linalg.eig(T_tilda)
    left = tf.linalg.adjoint(tf.linalg.inv(right))
    # eigendecomposition of T
    right = Y @ (v * lmbd_inv) @ right
    left = u @ left
    norm = tf.linalg.adjoint(left) * tf.linalg.matrix_transpose(right)
    norm = tf.reduce_sum(norm, axi=-1)
    norm = tf.math.sqrt(norm)
    right = right / norm
    left = left / tf.math.conj(norm)
    return eig_vals, right, left


def solve_regression(X, Y):
    """Solves the following linear regression problem
    ||TX - Y||_F --> min with respect to transition matrix T.
    T = Y @ pinv(X)
    Args:
        X: tensor of shape(n, ...)
        Y: tensor of shape(n, ...)
    Returns:
        tensor of shape (n, n), transition matrix
    Note:
        n -- dimension of one data point"""
    
    dtype = X.dtype
    s, u, v = tf.linalg.svd(X)
    ind = tf.cast(s > 1e-8, dtype=tf.int32)
    s_inv = tf.concat([1 / s[:ind], s[ind:]], axis=0)
    s_inv = tf.cast(s_inv, dtype=dtype)
    X_pinv = (v * s_inv) @ tf.linalg.adjoint(u)
    return Y @ X_pinv
