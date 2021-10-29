import numpy as np
import scipy as sp
from util import log_multivariate_normal, solve_triangular

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

def safe_inverse(C):
    try:
        return np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(C)

def constraint_matrices(M):
    d1, d2 = M.shape
    n = np.isnan(M).sum()
    M = M.flatten()
    D = np.zeros((d1 * d2, n))
    for r in range(M.shape[0]):
        if np.isnan(M[r]):
            c = r % n
            D[r][c] = 1.0
    
    f = np.nan_to_num(M, nan=0.0).flatten()
    return f, D

class KalmanFilter:
    def __init__(self, Phi, A, Q, R, x0=None, E0=None, constraints={}):
        self.Phi = Phi
        self.A = A
        self.Q = Q
        self.R = R
        self.x0 = x0 if type(x0) != type(None) else np.zeros(Q.shape[0])
        self.E0 = E0 if type(E0) != type(None) else Q
        self.constraints = constraints
        
#         # If R or Q is specified, their counterpart design matrix must
#         # also update in the constraint style            
#         if 'Q' in self.constraints and not 'Phi' in self.constraints:
#             self.constraints['Phi'] = constraint_matrices(np.full(Phi.shape, np.nan))

#         if 'R' in self.constraints and not 'A' in self.constraints:
#             self.constraints['A'] = constraint_matrices(np.full(A.shape, np.nan))

        for key in self.constraints:
            C = self.constraints[key]
            if type(C) == np.ndarray:
                self.constraints[key] = constraint_matrices(C)
                        
    def params(self):
        return self.Phi, self.A, self.Q, self.R, self.x0, self.E0

    def filter(self, y):
        Phi, A, Q, R, x0, E0 = self.params()

        xp = []
        xf = []
        ptf = []
        ptp = []
        ks = []

        xtt1 = x0
        Ptt1 = E0

        for i,v in enumerate(y):
            xp.append(xtt1)
            ptp.append(Ptt1)
            resid = v - A @ xtt1
            K = Ptt1 @ A.T @ np.linalg.pinv(A @ Ptt1 @ A.T + R)
            Ptt = (np.eye(Ptt1.shape[0]) - K @ A) @ Ptt1
            xtt = xtt1 + K @ resid
            
            xf.append(xtt)
            ptf.append(Ptt)
            ks.append(K)
            
            xtt1 = Phi @ xtt
            Ptt1 = (Phi @ Ptt @ Phi.T) + Q
        
        xp = np.array(xp)
        ptp = np.array(ptp)
        xf = np.array(xf)
        ptf = np.array(ptf)
        ks = np.array(ks)
        return xp, ptp, xf, ptf, ks

    def smooth(self, y):
        Phi, A, Q, R, x0, E0 = self.params()
        xp, ptp, xf, ptf, ks = self.filter(y)
        
        xs = [xf[-1]]
        pts = [ptf[-1]]
        ptt1s = []

        # Pn_n1_n = (I - K_n1 @ A) @ phi @ Pn_n1
        I = np.eye(ks[-1].shape[0])
        Pntt1 = (I - ks[-1] @ A) @ Phi @ ptf[-2]
        ptt1s.append(Pntt1)
            
        for i,v in enumerate(reversed(xf)):
            if i >= len(y) - 1:
                break

            xtt = xf[-(i + 2)]

            Ptt = ptf[-(i + 2)]
            Ptt1 = Phi @ Ptt @ Phi.T + Q
            J = Ptt @ Phi.T @ np.linalg.pinv(Ptt1)

            xnt = xtt + J @ (xs[-1] - Phi @ xtt)
            Pnt = Ptt + J @ (pts[-1] - Ptt1) @ J.T

            Pt1t1 = ptf[-(i + 2)]
            Pt1t = Phi @ Pt1t1 @ Phi.T + Q
            # J2 = Pt1t1 @ Phi.T @ np.linalg.pinv(Pt1t)
            # Pntt1 = Ptt @ J2 + J @ (Pntt1 - Phi @ Ptt) @ J2.T
            
            ptt1s.append(pts[-1] @ J.T)
            xs.append(xnt)
            pts.append(Pnt)

        ptt1s.append(np.zeros(pts[-1].shape))
        xs = np.array(list(reversed(xs)))
        pt = np.array(list(reversed(pts)))
        ptt1s = np.array(list(reversed(ptt1s[1:])))
        return xs, pt, ptt1s

    def log_likelihood(self, y):
        Phi, A, Q, R, x0, E0 = self.params()
        xp, ptp, xf, ptf, ks = self.filter(y)
        return log_multivariate_normal(y, A @ xp, A @ ptp @ A.T + R).sum()

    def expect(self, y):
        Phi, A, Q, R, x0, E0 = self.params()
        x, pt, ptt1 = self.smooth(y)
        xs = x[1:]
        s11 = (xs @ np.transpose(xs, axes=[0, 2, 1]) + pt[1:])
        s10 = (xs @ np.transpose(x[:-1], axes=[0, 2, 1]) + ptt1[1:])
        s00 = (x[:-1] @ np.transpose(x[:-1], axes=[0, 2, 1]) + pt[:-1])
        e = y[1:] - (A @ x[1:])
        return x, pt, ptt1, s11, s10, s00, e
        
    def em_iter(self, y, em_vars=['Phi', 'Q', 'R', 'x0', 'E0'], strict=False):
        Phi, A, Q, R, x0, E0 = self.params()
        i = 0
        result = None

        def expect():
            nonlocal i, result
            
            if i == 0 or strict == True:
                result = self.expect(y)

            i += 1
            return result

        if 'Phi' in em_vars:
            x, pt, ptt1, s11, s10, s00, e = expect()
            
            if 'Phi' in self.constraints:
                f, D = self.constraints['Phi']
                qk = np.kron(Q, s00)
                Phi1 = (Q @ s10).reshape((s10.shape[0], Q.shape[1] ** 2))
                Phi1 -= qk @ f
                Phi1 = Phi1 @ D
                Phi2 = D.T @ qk @ D
                phi = np.linalg.inv(Phi2.sum(axis=0)) @ Phi1.sum(axis=0)
                phi = f + D @ phi
                self.Phi = Phi = phi.reshape(Phi.shape)
            else:
                s00inv = safe_inverse(s00.sum(axis=0))
                self.Phi = Phi = s10.sum(axis=0) @ s00inv
                            
        if 'Q' in em_vars:
            x, pt, ptt1, s11, s10, s00, e = expect()
            Q = (s11 - s10 @ Phi.T - Phi @ np.transpose(s10, axes=[0, 2, 1]) + Phi @ s00 @ Phi.T)
            
            if 'Q' in self.constraints:
                f, D = self.constraints['Q']
                q = Q.reshape((Q.shape[0], Q.shape[1] ** 2))
                q = np.linalg.inv(Q.shape[0] * D.T @ D) @ (q @ D).sum(axis=0)
                q = f + D @ q
                self.Q = Q = q.reshape(Q.shape[1:])
            else:
                self.Q = Q = Q.mean(axis=0)
            
        if 'x0' in em_vars:
            x, pt, ptt1, s11, s10, s00, e = expect()
            self.x0 = x0 = x[0]

        if 'E0' in em_vars:
            x, pt, ptt1, s11, s10, s00, e = expect()
            self.E0 = E0 = pt[0]

        if 'A' in em_vars:
            x, pt, ptt1, s11, s10, s00, e = expect()
            if 'A' in self.constraints:
                f, D = self.constraints['A']
                rk = np.kron(R, pt + x @ np.transpose(x, axes=[0, 2, 1]))
                
                A1 = (R @ y @ np.transpose(x, axes=[0, 2, 1]))
                A1 = A1.reshape((A1.shape[0], R.shape[0] * Q.shape[0]))
                A1 -= rk @ f
                A1 = A1 @ D
                
                A2 = D.T @ rk @ D
                
                a = np.linalg.inv(A2.sum(axis=0)) @ A1.sum(axis=0)
                self.A = A = (f + D @ a).reshape(A.shape)
            else:
                A1 = (y @ np.transpose(x, axes=[0, 2, 1])).sum(axis=0)
                A2 = np.linalg.inv((pt + x @ np.transpose(x, axes=[0, 2, 1])).sum(axis=0))
                self.A = A = A1 @ A2
                        
        if 'R' in em_vars:
            x, pt, ptt1, s11, s10, s00, e = expect()
            resid = (e @ np.transpose(e, axes=[0, 2, 1]))
            va = (A @ pt[1:] @ A.T)
            R = (resid + va)
            
            if 'R' in self.constraints:
                f, D = self.constraints['R']
                r = R.reshape((R.shape[0], R.shape[1] ** 2))
                r = np.linalg.inv(R.shape[0] * D.T @ D) @ (r @ D).sum(axis=0)
                r = f + D @ r
                self.R = R = r.reshape(R.shape[1:])
            else:
                self.R = R = R.mean(axis=0)
            
    def em(self, y, n=10, **kwargs):
        print('\t[ll: {:.2f}]'.format(self.log_likelihood(y)))

        for i in range(n):
            self.em_iter(y, **kwargs)
            ll = self.log_likelihood(y)
            print('\t[ll: {:.2f}]'.format(ll))
            if np.isnan(ll):
                raise ValueError('nan encountered')



def unflatten_kalman_params(params, state_dim, obs_dim, args):
    i = 0
    
    if 'Phi' in args:
        Phi = args['Phi']
    else:
        if 'DiagPhi' in args and args['DiagPhi'] == True:
            Phi = np.diagflat(params[i:i+state_dim])
            i += state_dim
        else:
            Phi = params[i:i+state_dim**2].reshape((state_dim, state_dim))
            i += state_dim**2
    
    if 'A' in args:
        A = args['A']
    else:
        dim = obs_dim * state_dim
        A = params[i:i+dim].reshape((obs_dim, state_dim))
        i += dim

    qp = (state_dim * (state_dim - 1)) // 2 + state_dim    
    if 'Q' in args:
        Q = args['Q']
    else:
        if 'DiagQ' in args and args['DiagQ'] == True:
            Q = np.diagflat(np.diag(params[i:i+state_dim]))
            i += state_dim
        else:
            Q = cholesky_unflatten(params[i:i+qp], state_dim)
            i += qp

    if 'R' in args:
        R = args['R']
    else:
        if 'DiagR' in args and args['DiagR'] == True:
            R = np.diagflat(np.diag(params[i:i+obs_dim]))
            i += obs_dim
        else:
            rp = (obs_dim * (obs_dim - 1)) // 2 + obs_dim
            R = cholesky_unflatten(params[i:i+rp], obs_dim)
            i += rp
   
    if 'x0' in args:
        x0 = args['x0']
    else:
        x0 = params[i:i+state_dim].reshape((state_dim, 1))
        i += state_dim

    if 'E0' in args:
        E0 = args['E0']
    else:
        E0 = construct_cov(params[i:i+qp], state_dim)
        i += qp
        
    return Phi, A, Q, R, x0, E0
    
def kalman_likelihood(params, args, state_dim, y):
    Phi, A, Q, R, x0, E0 = unflatten_kalman_params(
        params, 
        state_dim, 
        y.shape[1], 
        args
    )
    
    kf = KalmanFilter(
        Phi,
        A,
        Q,
        R,
        x0,
        E0
    )
    
    ll = -kf.log_likelihood(y)
    print(ll)
    return ll