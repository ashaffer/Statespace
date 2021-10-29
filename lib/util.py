import numpy as np
import scipy as sp

def cov_logdet(C, cholesky=False, min_covar=1.e-7):
    if cholesky == False:
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(C + min_covar * np.eye(C.shape[-1]))
    else:
        L = C

    # Use negative axes so that this works smoothly with tensors
    # that contain main covariance mtarices
    D = np.diagonal(L, axis1=-2, axis2=-1) 
    return 2 * np.log(D)

def solve_triangular(L, Y):
    if L.ndim == 2:
        return sp.linalg.solve_triangular(L, Y.T, lower=True, check_finite=False)
    
    arr = []
    
    for i in range(L.shape[0]):
        arr.append(sp.linalg.solve_triangular(L[i], Y[i].T, lower=True, check_finite=False))
    
    return np.array(arr)

def log_multivariate_normal(x, mu, C, min_covar=1.e-7):
    try:
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Occasionally we can end up with very tiny negative values on the diagonal
            # for numerical reasons, and these can make an otherwise positive definite
            # covariance matrix negative, which ruins everything, so add a tiny amount of 
            # positivity.
            L = min_covar * np.eye(C.shape[-1])

        cld = cov_logdet(L, cholesky=True)
        S = solve_triangular(L, np.transpose(x - mu, axes=[0, 2, 1]))
        S = S.sum(axis=2)
        return -0.5 * (np.log(2 * np.pi) + cld + S ** 2)
    except np.linalg.LinAlgError:
        print('Warning: Using slower covariance path')
        # Our covariance is likely not positive definite, which can happen when it isn't
        # full rank
        cld = np.log(np.linalg.eigvalsh(C)).sum()
        if np.isnan(cld):
            print('nan', C)
        resid = (x - mu)
        err = resid @ np.linalg.pinv(C) @ resid.T
        # p = np.log(cld * np.exp(err))
        return -0.5 * (np.log(2 * np.pi) + cld + err)
    
def construct_cov(p, dim):
    C = np.zeros((dim, dim))
    C[np.triu_indices(dim)] = p
    D = np.diagflat(np.diag(C))
    C += C.T
    C -= D
    return C

def cholesky_flatten(C, min_covar=1.e-7):
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(C + min_covar * np.eye(C.shape[-1]))
    
    return L[np.tril_indices(L.shape[-1])]

def cholesky_unflatten(p, dim):
    C = np.zeros((dim, dim))
    C[np.tril_indices(dim)] = p
    return C @ C.T