import numpy as np
import scipy as sp
from . import util

class KalmanFilter:
    def __init__(self, Phi, A, Q, R, x0=None, E0=None, constraints={}):
        self.Phi = Phi
        self.A = A
        self.Q = Q
        self.R = R
        self.x0 = x0 if type(x0) != type(None) else np.zeros((Q.shape[0], 1))
        self.E0 = E0 if type(E0) != type(None) else Q
        self.constraints = constraints

        params = {
            'Phi': self.Phi,
            'A': self.A,
            'Q': self.Q,
            'R': self.R,
            'x0': self.x0,
            'E0': self.E0
        }
        
        for key in self.constraints:
            C = self.constraints[key]
            if type(C) == str:
                C = util.constraint_str_to_matrix(C, params[key].shape)

            if type(C) == np.ndarray:
                self.constraints[key] = util.constraint_matrices(C)
                     
    def named_params(self):
        return {
            'Phi': self.Phi,
            'A': self.A,
            'Q': self.Q,
            'R': self.R,
            'x0': self.x0,
            'E0': self.E0
        }

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

    def log_likelihood(self, y, params=None):
        params = self.params() if params == None else params
        Phi, A, Q, R, x0, E0 = params
        xp, ptp, xf, ptf, ks = self.filter(y)
        return util.log_multivariate_normal(y, A @ xp, A @ ptp @ A.T + R).sum()

    def expect(self, y):
        Phi, A, Q, R, x0, E0 = self.params()
        x, pt, ptt1 = self.smooth(y)
        xs = x[1:]
        s11 = (xs @ np.transpose(xs, axes=[0, 2, 1]) + pt[1:])
        s10 = (xs @ np.transpose(x[:-1], axes=[0, 2, 1]) + ptt1[1:])
        s00 = (x[:-1] @ np.transpose(x[:-1], axes=[0, 2, 1]) + pt[:-1])
        e = y[1:] - (A @ x[1:])
        return x, pt, ptt1, s11, s10, s00, e
        
    def em_iter(self, y, em_vars=['Phi', 'Q', 'A', 'R', 'x0', 'E0'], strict=False):
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
                Qi = np.linalg.inv(Q)
                f, D = self.constraints['Phi']
                qk = np.kron(Qi, s00)
                Phi1 = (Qi @ s10).reshape((s10.shape[0], Q.shape[1] ** 2))
                Phi1 -= qk @ f
                Phi1 = Phi1 @ D
                Phi2 = D.T @ qk @ D
                phi = np.linalg.inv(Phi2.sum(axis=0)) @ Phi1.sum(axis=0)
                phi = f + D @ phi
                self.Phi = Phi = phi.reshape(Phi.shape)
            else:
                s00inv = util.safe_inverse(s00.sum(axis=0))
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
                Ri = np.linalg.inv(R)

                f, D = self.constraints['A']
                # This is the reverse of the argument order used in the paper, but
                # passing them the other way doesn't seem to work.
                rk = np.kron(Ri, pt + x @ np.transpose(x, axes=[0, 2, 1]))
                
                A1 = (Ri @ (y @ np.transpose(x, axes=[0, 2, 1])))
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
        print('\t[0] ll: {:.2f}'.format(self.log_likelihood(y)))

        for i in range(n):
            self.em_iter(y, **kwargs)
            if (i + 1) % 10 == 0:
                ll = self.log_likelihood(y)
                print('\t[{}] ll: {:.2f}'.format(i + 1, ll))
                if np.isnan(ll):
                    raise ValueError('nan encountered')

        # If we ended on a multiple o 10, the last ll was already printed
        if n % 10 != 0:
            ll = self.log_likelihood(y)
            print('[{}] ll: {:.2f}'.format(n, ll))

    def minimize(self, y, optimize):
        p = self.named_params()
        args = []

        for name in optimize:
            if name in self.constraints:
                f, D = self.constraints[name]
                param = np.linalg.pinv(D) @ (p[name].flatten() - f)
                args.append(param)
            else:
                args.append(p[name].flatten())

        constants = {}
        for name in p:
            if name not in optimize:
                constants[name] = p[name]

        r = sp.optimize.minimize(
            kalman_likelihood,
            np.concatenate(args),
            (
                constants,
                self.Q.shape[0],
                y,
                self.constraints
            )
        )

        Phi, A, Q, R, x0, E0 = unflatten_kalman_params(
            r.x, 
            self.Q.shape[0], 
            y.shape[1], 
            constants,
            self.constraints
        )

        self.Phi = Phi
        self.A = A
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.E0 = E0
        print('Minimized: {:.2f}'.format(self.log_likelihood(y)))

def extract_parameter(i, params, shape, constraint, cholesky=False):
    if constraint != None:
        f, D = constraint
        P = params[i:i+D.shape[1]]
        i += D.shape[1]
        L = util.constrained_cholesky(f, D, P)
        P = L @ L.T
    elif cholesky == True:
        n = (shape[0] * (shape[1] - 1)) // 2 + shape[0]
        P = params[i:i+n]
        i += n
        P = util.cholesky_unflatten(P, shape[0])
    else:
        n = shape[0] * shape[1]
        P = params[i:i+n]
        P = P.reshape(shape)
        i += n

    return P, i

def unflatten_kalman_params(params, state_dim, obs_dim, args, constraints):
    i = 0

    if 'Phi' in args:
        Phi = args['Phi']
    else:
        Phi, i = extract_parameter(i, params, (state_dim, state_dim), constraints['Phi'] if 'Phi' in constraints else None)
    
    if 'A' in args:
        A = args['A']
    else:
        A, i = extract_parameter(i, params, (obs_dim, state_dim), constraints['A'] if 'A' in constraints else None)

    if 'Q' in args:
        Q = args['Q']
    else:
        Q, i = extract_parameter(i, params, (state_dim, state_dim), constraints['Q'] if 'Q' in constraints else None, cholesky=True)

    if 'R' in args:
        R = args['R']
    else:
        R, i = extract_parameter(i, params, (obs_dim, obs_dim), constraints['R'] if 'R' in constraints else None, cholesky=True)
   
    if 'x0' in args:
        x0 = args['x0']
    else:
        x0, i = extract_parameter(i, params, (state_dim, 1), None)

    if 'E0' in args:
        E0 = args['E0']
    else:
        E0, i = extract_parameter(i, params, (state_dim, state_dim), None, cholesky=True)
        
    return Phi, A, Q, R, x0, E0
    
def kalman_likelihood(params, args, state_dim, y, constraints):
    Phi, A, Q, R, x0, E0 = unflatten_kalman_params(
        params, 
        state_dim, 
        y.shape[1],
        args,
        constraints
    )
    
    kf = KalmanFilter(
        Phi,
        A,
        Q,
        R,
        x0,
        E0
    )
    
    return -kf.log_likelihood(y)