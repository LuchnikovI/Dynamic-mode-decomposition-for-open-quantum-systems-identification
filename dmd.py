import tensorflow as tf

def dmd(X, Y, eps=1e-5):
    """Solves the following linear regression problem
    ||TX - Y||_F --> min with respect to transition matrix T.
    Matrix T is found by using dynamic mode decomposition (dmd) in the form
    of its eigendecomposition with the minimal possible rank.
    You may read more about dmd in the following paper
    https://arxiv.org/pdf/1312.0041.pdf
    Args:
        X: tensor of shape(..., n)
        Y: tensor of shape(..., n)
        eps: float value, tolerance that defines rank
    Returns:
        three tensors of shapes (r,), (r, r), and (r, r),
        dominant eigenvalues and corresponding (right and left)
        eigenvectors"""

    X_resh = tf.reshape(X, (X.shape[-1], -1))
    Y_resh = tf.reshape(Y, (Y.shape[-1], -1))
    # SVD of X_resh matrix
    lmbd, u, v = tf.linalg.svd(X_resh)
    # number of singular vals > eps
    ind = tf.reduce_sum(tf.cast(lmbd > eps, dtype=tf.int32))
    # truncation of all elements of the svd
    lmbd = lmbd[:ind]
    lmbd_inv = 1 / lmbd
    lmbd_inv = tf.linalg.diag(lmbd_inv)
    lmbd_inv = tf.cast(lmbd_inv, dtype=tf.complex128)
    u = u[:, :ind]
    v = v[:, :ind]
    # eigendecomposition of T_tilda
    T_tilda = tf.linalg.adjoint(u) @ Y_resh @ v @ lmbd_inv
    eig_vals, right = tf.linalg.eig(T_tilda)
    left = tf.linalg.adjoint(tf.linalg.inv(right))
    right = Y @ v @ lmbd_inv @ right
    left = u @ left
    norm = tf.linalg.diag_part(tf.linalg.adjoint(left) @ right)
    norm = tf.math.sqrt(norm)
    right = right / norm
    left = left / tf.math.conj(norm)
    return eig_vals, right, left