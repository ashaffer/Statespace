import numpy as np
import scipy as sp

def cov_logdet(C, cholesky=False, min_covar=1.e-7):
    if cholesky == False:
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            print('cov_logdet cholesky failed')
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
        print('cholesky failed')
        L = np.linalg.cholesky(C + min_covar * np.eye(C.shape[-1]))
    
    return L[np.tril_indices(L.shape[-1])]

def cholesky_unflatten(p, dim):
    C = np.zeros((dim, dim))
    C[np.tril_indices(dim)] = p
    return C @ C.T

def safe_inverse(C):
    try:
        return np.linalg.inv(C)
    except np.linalg.LinAlgError:
        print('inverse failed')
        return np.linalg.pinv(C)

def constraint_str_to_matrix(s, dims):
    if s == 'lower_triangular_s1':
        M = np.full(dims, np.nan)
        for i in range(dims[1] - 1):
            for j in range(dims[1]):
                if j > i:
                    M[i][j] = 0.0

    return M

def constraint_matrices(M):
    d1, d2 = M.shape
    n = np.isnan(M).sum()
    M = M.flatten()
    D = np.zeros((d1 * d2, n))
    offset = 0

    for r in range(d1 * d2):
        if np.isnan(M[r]):
            c = (r - offset) % n
            D[r][c] = 1.0
        else:
            offset += 1
    
    f = np.nan_to_num(M, nan=0.0).flatten()
    return f, D


def cholesky_value(L, i, j, c):
    if i == j:
        return np.sqrt(c - sum(L[j][k] ** 2 for k in range(i)))
    else:
        return (c - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]

def constrained_cholesky(f, D, r):
    dim = int(np.sqrt(f.shape[0]))
    L = np.zeros((dim, dim))
    v = np.zeros(r.shape)
    z = np.zeros(r.shape)
    
    for p in range(f.shape[0]):
        i = p // dim
        j = p % dim
        if j > i:
            continue
            
        if np.all(D[p] == 0.0):
            # C just has a defined constant in this position
            L[i][j] = cholesky_value(L, i, j, f[p])
        else:
            used = False
            for k in range(D.shape[1]):
                if D[p][k] != 0.0 and v[k] == 0.0:
                    # While this variable is still undetermined,
                    # pull out the next unused rv from the vector
                    v[k] = 1.0
                    L[i][j] = f[p] + r[k]
                    used = True
                    break
            
            if not used:
                # This variable has already been fully determined by prior
                # iterations, so we need to calculate what values those prior
                # choices imply, and then run those values forward to get 
                # what value of C they imply for our covariance matrix.
                Di = D.copy()
                Di[p:] = 0.0
                Di = np.linalg.pinv(Di)
                z = Di @ ((L @ L.T).flatten() - f)

                C = f + D @ z
                L[i][j] = cholesky_value(L, i, j, C[p])
    return L