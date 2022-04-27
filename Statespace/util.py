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
        arr.append(sp.linalg.solve_triangular(L[i], Y[i], lower=True, check_finite=False))
    
    return np.array(arr)

def multivariate_normal_density(x, mu, C):
    resid = x - mu
    e = -0.5 * resid.T @ np.linalg.inv(C) @ resid
    n = 1 if np.isscalar(x) else len(x)
    return np.exp(e) / np.sqrt((2 * np.pi) ** n * np.linalg.det(C))

def normal_density(x, mu, s2):
    e = (x - mu) ** 2 / (2 * s2)
    return np.exp(-e) / np.sqrt(2 * np.pi * s2)

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
        xc = x - mu

        if xc.ndim == 2:
            # The right hand side of the equation needs to match the dimensionality
            # of the L matrix
            xc = np.expand_dims(xc, [2])
            xc = np.transpose(xc, axes=[0, 2, 1])

        S = solve_triangular(L, xc)
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
        L = safe_cholesky(C)
    except np.linalg.LinAlgError:
        print('cholesky failed')
        L = np.linalg.cholesky(C + min_covar * np.eye(C.shape[-1]))
    
    # L[np.diag_indices(C.shape[-1])] = np.log(np.diag(L))
    return L[np.tril_indices(L.shape[-1])][:,None]

def cholesky_unflatten(p, dim):
    C = np.zeros((dim, dim))
    C[np.tril_indices(dim)] = p[:,0]
    # C[np.diag_indices(dim)] = np.exp(np.diag(C)) - 1
    return C @ C.T

def safe_cholesky(C):
    d = np.diag(C)
    z = d == 0.0

    if z.any():
        E = row_extractor(C.shape[0], ~z)
        Inv = safe_cholesky(E @ C @ E.T)
        return E.T @ Inv @ E

    return np.linalg.cholesky(C)

def safe_inverse(C):
    if C.ndim == 3:
        return np.array([
            safe_inverse(C[i]) for i in range(C.shape[0])
        ])

    d = np.diag(C)
    z = d == 0.0

    if z.any():
        E = row_extractor(C.shape[0], ~z)
        Inv = safe_inverse(E @ C @ E.T)
        return E.T @ Inv @ E
    else:
        try:
            return np.linalg.inv(C)
        except np.linalg.LinAlgError:
            # raise ValueError('tmp')
            print('Inverse failed')
            print(C)
            return np.linalg.pinv(C)

def constraint_matrices(M):
    if M.ndim == 1:
        M = M[:,None]

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
    
    f = np.nan_to_num(M, nan=0.0).flatten()[:,None]

    # If we have no free variables, we still need D to exist as a thing
    # we can multiply, it just should be all zeros.
    if D.shape[1] == 0:
        D = np.zeros((D.shape[0], 1))

    return f, D


def cholesky_value(L, i, j, c):
    if i == j:
        return np.sqrt(c - sum(L[j][k] ** 2 for k in range(i)))
    else:
        if L[j][j] == 0.0:
            return 0.0
        else:
            return (c - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]

def constrained_cholesky(f, D, r, eps=1e-8):
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
                    if i != j:
                        # If we have an offset for this value, we need to divide
                        # it by L[j]j] as it would have originally been                        
                        if L[j][j] == 0.0:
                            L[i][j] = 0.0
                        else:
                            L[i][j] = f[p] / L[j][j] + r[k]
                    else:
                        # The floor for this value is f[p] minus the squared sum
                        # of the prior diagonals. This minimum value would produce
                        # a diagonal of exactly f[p] if nothing was added to it.
                        a = f[p] - sum(L[j][m] ** 2 for m in range(i))
                        # Ensure that this value is always at least epsilon
                        # above zero, since it's a variance
                        L[i][j] = np.sqrt(max(a, 0)) + np.exp(r[k]) - (1 - eps)
 
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

def inverse_constrained_cholesky(f, D, L, eps=1e-8):
    Lp = np.full(L.shape, 0.0)
    dim = int(np.sqrt(f.shape[0]))

    for p in range(f.shape[0]):
        i = p // dim
        j = p % dim
        if j > i:
            continue

        if np.any(D[p] != 0.0):
            if i != j:
                Lp[i][j] = L[i][j]
            else:
                Lp[i][j] = np.log(L[i][j] + (1 - eps))

    return np.linalg.pinv(D) @ Lp.flatten()

def validate_cholesky(f, D, C):
    v = inverse_constrained_cholesky(f, D, safe_cholesky(C))[:,None]
    L2 = constrained_cholesky(f, D, v)
    C2 = L2 @ L2.T
    if not np.allclose(C, C2):
        print('invalid constraint inverse')

def sqrtm(M):
    if M.ndim == 3:
        return np.array([sqrtm(M[i]) for i in range(M.shape[0])])
    else:
        return sp.linalg.sqrtm(M)

def solve_stationary(A):
    A = A - np.eye(A.shape[0])
    A[-1] = np.ones(A.shape[0])
    b = np.zeros(A.shape[0])
    b[-1] = 1.0
    return sp.linalg.solve(A, b, transposed=False)

def roll_tile(x, n):
    b = x.dtype.itemsize
    r = np.lib.stride_tricks.as_strided(x, shape=(x.shape[0], n), strides=(b, b))
    return r[:len(r) - (n - 1)]


def row_extractor(d, rows):
    # Also allow rows to be passed in as a list of indices
    if len(rows) != d:
        rows = [i in rows for i in range(d)]

    M = np.zeros((sum(rows), d))
    j = 0

    for i in range(d):
        if rows[i]:
            M[j][i] = 1.0
            j += 1

    return M

def to_adjacency(M):
    M = np.copy(M)
    M[M != 0.0] = 1.0
    return M

def reachable_edges(Adj, nodes):
    I = np.eye(len(nodes))
    S = np.diagflat(nodes)

    clean = [I - S]
    touched = [np.zeros(clean[-1].shape)]
    M = Adj
        
    for i in range(Adj.shape[0]):
        T = to_adjacency(np.diagflat(M @ nodes))
        D = I - T
        clean.append(I - T)
        touched.append(T - S)
        M = M @ Adj
    
    return clean, touched, S

def extend_matrices(n, Ms):
    Ms = np.array(Ms)
    d = n - Ms.shape[0]
    Ms = np.concatenate((Ms, np.broadcast_to(Ms[-1], (d, *Ms[-1].shape))), axis=0)
    return Ms

def is_number_type(x):
    t = type(x)
    return (
           t == int
        or t == float
        or isinstance(x, np.floating)
        or isinstance(x, np.integer)
    )

def symm(M):
    return 0.5 * (M + M.T)
