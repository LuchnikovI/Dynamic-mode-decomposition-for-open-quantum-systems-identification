import tensorflow as tf
import math

@tf.function
def f_basis(n, dtype=tf.complex128):
    """The function returns orthonormal basis in the linear space of
    real traceless matrices of size n. For all matrices,
    the following condition holds true <F_i, F_j> = I_ij,
    where I is the identity matrix.
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
        K: int value, memory depth
    Returns:
        tensor of shape (batch_size, n-K+1, K, m)"""

    shape_inv = tf.TensorShape([T.get_shape()[0],
                                None,
                                K,
                                T.get_shape()[-1]])

    L = T.shape[1]
    i = tf.constant(1)
    t = T[:, tf.newaxis, :K]
    cond = lambda i, t: i<=L-K
    body = lambda i, t: [i+1, tf.concat([t, T[:, tf.newaxis, i:K+i]], axis=1)]
    _, t = tf.while_loop(cond, body, loop_vars=[i, t],
                  shape_invariants=[i.get_shape(), shape_inv])
    return t


def trunc_svd(X, eps=1e-6):
    """Calculates truncated svd of a matrix with a given std of noise.
    Args:
        X: complex valued tensor of shape (q, p)
        eps: real valued scalar, std of additive noise
    Returns:
        three complex valued tensors u, s, v of shapes (q, r),
        (r,), (p, r), where r is optimal rank"""

    # svd
    s, u, v = tf.linalg.svd(X)

    real_dtype = s.dtype
    complex_dtype = u.dtype
    shape = tf.shape(X)

    # threshold
    q, p = shape[0], shape[1]
    q, p = tf.cast(q, dtype=real_dtype), tf.cast(p, dtype=real_dtype)
    threshold = eps * (tf.math.sqrt(2 * q) + tf.math.sqrt(2 * p))

    # optimal rank
    r = tf.reduce_sum(tf.cast(s > threshold, dtype=tf.int32))
    return tf.cast(s[:r], dtype=complex_dtype), u[:, :r], v[:, :r]


def optimal_K(trajectories, eps=1e-6):
    """Returns minimal sufficient K.
    Args:
        trajectories: complex valued tensor of shape (bs, n, m),
            quantum trajectories, bs enumerates trajectories, n is total
            number of time steps, m is dimension of density matrix
        eps: std of additive noise
    Returns:
        int number, minimal sufficient K"""

    shape = tf.shape(trajectories)
    bs, n,  m = shape[0], shape[1], shape[2]
    dtype = trajectories.dtype
    
    # initial parameters
    K = tf.constant(0)
    err = tf.math.real(tf.constant(1e8, dtype=dtype))
    q = tf.constant(0)
    p = tf.constant(0)
    
    def body(K, err, q, p):
        K_new = K + 1
        H = hankel(trajectories, int(K_new))
        N = n - K_new
        X = H[:, :-1]
        Y = H[:, 1:]
        q = tf.constant(bs * N)
        p = tf.constant(K_new * m)
        X_resh = tf.reshape(X, (q, p))
        Y_resh = tf.reshape(Y, (q, p))
        X_resh = tf.transpose(X_resh)
        Y_resh = tf.transpose(Y_resh)
        _, _, v = trunc_svd(X_resh, eps=eps)
        delta = Y_resh - Y_resh @ v @ tf.linalg.adjoint(v)
        return K_new, tf.math.real(tf.linalg.norm(delta)), q, p

    cond = lambda K, err, q, p: err > 0.9 * tf.math.sqrt(tf.cast(2 * q * p, dtype=err.dtype)) * eps

    K, _, _, _ = tf.while_loop(cond, body, loop_vars=[K, err, q, p])
    return int(K)


def dmd(trajectories, K=None, eps=1e-6, auto_K=False, type='exact'):
    """Solves the following linear regression problem
    ||TX - Y||_F --> min with respect to transition matrix T.
    Matrix T is found by using dynamic mode decomposition (dmd) in the form
    of its eigendecomposition with the minimal possible rank.
    You may read more about dmd in the following paper
    https://arxiv.org/pdf/1312.0041.pdf
    Args:
        trajectories: complex valued tensor of shape (bs, n, m, m),
            quantum trajectories, bs enumerates trajectories, n is total
            number of time steps, m is dimension of density matrix
        K: int value, memory depth
        eps: float value, std of additive noise
        auto_K: boolean value, shows if we use automatic K determination
            or not
        type: string specifying type of DMD ('standard' or 'exact')
    Returns:
        three tensors of shapes (r,), (n, r), and (n, r),
        dominant eigenvalues and corresponding (right and left)
        eigenvectors, and one int value, representing the minimal sufficient K
    Note:
        n -- dimension of one data point, r -- rank that is determined
        by tolerance eps."""
    
    # bs is batch size
    # n is number of time steps
    # m is the size of density matrix
    bs, n, m, _ = trajectories.shape
    # reshape density matrices to vectors
    t = tf.reshape(trajectories, (bs, n, m**2))
    # build hankel matrix of shape (bs, n-K+1, K, m**2)
    if auto_K:
        K_opt = optimal_K(t, eps=eps)
    else:
        K_opt = K
    t = hankel(t, K_opt)
    # build X and Y tensors, both have shape (K*(m**2), bs, n-K)
    t = tf.reshape(t, (bs, n-K_opt+1, K_opt*(m**2)))
    t = tf.transpose(t, (2, 0, 1))
    X = t[..., :-1]
    Y = t[..., 1:]
    # reshape X and Y tensors to matrices
    X_resh = tf.reshape(X, (K_opt*(m**2), bs*(n-K_opt)))
    Y_resh = tf.reshape(Y, (K_opt*(m**2), bs*(n-K_opt)))
    # SVD of X_resh matrix
    lmbd, u, v = trunc_svd(X_resh, eps=eps)
    # inverse of singular values
    lmbd_inv = 1 / lmbd
    # eigendecomposition of T_tilda
    T_tilda = tf.linalg.adjoint(u) @ Y_resh @ (v * lmbd_inv)
    eig_vals, right = tf.linalg.eig(T_tilda)
    left = tf.linalg.adjoint(tf.linalg.inv(right))
    # eigendecomposition of T
    if type == 'standard':
        right = u @ right
    elif type == 'exact':
        right = Y_resh @ (v * lmbd_inv) @ right
    left = u @ left
    norm = tf.linalg.adjoint(left) * tf.linalg.matrix_transpose(right)
    norm = tf.reduce_sum(norm, axis=-1)
    norm = tf.math.sqrt(norm)
    right = right / norm
    left = left / tf.math.conj(norm)
    return eig_vals, right, left, K_opt


@tf.function
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
    X_resh = tf.reshape(X, (X.shape[0], -1))
    Y_resh = tf.reshape(Y, (Y.shape[0], -1))
    s, u, v = tf.linalg.svd(X_resh)
    ind = tf.cast(s > 1e-8, dtype=tf.int32)
    ind = tf.reduce_sum(ind)
    s_inv = tf.concat([1 / s[:ind], s[ind:]], axis=0)
    s_inv = tf.cast(s_inv, dtype=dtype)
    X_pinv = (v * s_inv) @ tf.linalg.adjoint(u)
    return Y_resh @ X_pinv
