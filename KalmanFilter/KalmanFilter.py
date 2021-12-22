import numpy as np
import scipy as sp
from . import util

class KalmanFilter:
    def __init__(self, Phi, A, Q, R, x0=None, E0=None, C=None, U=None, constraints={}):
        self.Phi = Phi
        self.A = A
        self.Q = Q
        self.R = R
        self.x0 = x0 if type(x0) != type(None) else np.zeros((Q.shape[0], 1))
        self.E0 = E0 if type(E0) != type(None) else Q
        self.C = np.zeros((Q.shape[0], 1)) if type(C) == type(None) else C
        self.U = np.zeros((R.shape[0], 1)) if type(U) == type(None) else U
        self.constraints = constraints

        params = {
            'Phi': self.Phi,
            'A': self.A,
            'Q': self.Q,
            'R': self.R,
            'x0': self.x0,
            'E0': self.E0,
            'C': self.C,
            'U': self.U
        }
        
        if 'C' in self.constraints:
            raise ValueError('Constraints on C are not allowed')

        if 'U' in self.constraints:
            raise ValueError('Constraints on U are not allowed')

        for key in self.constraints:
            C = self.constraints[key]
            if type(C) == str:
                C = util.constraint_str_to_matrix(C, params[key].shape)

            if type(C) == np.ndarray:
                self.constraints[key] = util.constraint_matrices(C)
                     
    def default_state_exog(self, y):
        return np.ones((y.shape[0], 1, 1))
    
    def default_obs_exog(self, y):
        return np.ones((y.shape[0], 1, 1))

    def named_params(self):
        return {
            'Phi': self.Phi,
            'A': self.A,
            'C': self.C,
            'Q': self.Q,
            'R': self.R,
            'U': self.U,
            'x0': self.x0,
            'E0': self.E0,
        }

    def params(self):
        return self.Phi, self.A, self.C, self.Q, self.R, self.U, self.x0, self.E0

    def filter(self, y, state_exog=None, obs_exog=None):
        if type(state_exog) == type(None):
            state_exog = self.default_state_exog(y)
        if type(obs_exog) == type(None):
            obs_exog = self.default_obs_exog(y)

        Phi, A, C, Q, R, U, x0, E0 = self.params()

        xp = []
        xf = []
        ptf = []
        ptp = []
        ks = []

        # Pre-compute the state and obs intercepts for the whole series
        xa = (C @ state_exog.T).T
        ya = (U @ obs_exog.T).T
        xtt = x0
        Ptt = E0

        for i,v in enumerate(y):
            Phii = Phi if Phi.ndim == 2 else Phi[i]
            Qi = Q if Q.ndim == 2 else Q[i]
            Ai = A if A.ndim == 2 else A[i]
            Ri = R if R.ndim == 2 else R[i]

            xtt1 = Phii @ xtt + xa[i]
            Ptt1 = (Phii @ Ptt @ Phii.T) + Qi

            xp.append(xtt1)
            ptp.append(Ptt1)
            
            resid = v - Ai @ xtt1 - ya[i]
            K = Ptt1 @ Ai.T @ np.linalg.pinv(Ai @ Ptt1 @ Ai.T + Ri)
            Ptt = (np.eye(Ptt1.shape[0]) - K @ Ai) @ Ptt1
            xtt = xtt1 + K @ resid

            xf.append(xtt)
            ptf.append(Ptt)
            ks.append(K)
        
        xp = np.array(xp)
        ptp = np.array(ptp)
        xf = np.array(xf)
        ptf = np.array(ptf)
        ks = np.array(ks)
        return xp, ptp, xf, ptf, ks

    def smooth(self, y, state_exog=None, obs_exog=None):
        Phi, A, C, Q, R, U, x0, E0 = self.params()
        xp, ptp, xf, ptf, ks = self.filter(y, state_exog=state_exog, obs_exog=obs_exog)
        
        xs = [xf[-1]]
        pts = [ptf[-1]]
        ptt1s = []

        Ai = A if A.ndim == 2 else A[-1]
        Phii = Phi if Phi.ndim == 2 else Phi[-1]

        # Pn_n1_n = (I - K_n1 @ A) @ phi @ Pn_n1
        I = np.eye(ks[-1].shape[0])
        Pntt1 = (I - ks[-1] @ Ai) @ Phii @ ptf[-2]
        ptt1s.append(Pntt1)
            
        for i,v in enumerate(reversed(xf)):
            Phii = Phi if Phi.ndim == 2 else Phi[i]
            Qi = Q if Q.ndim == 2 else Q[i]

            Ptt = ptf[-(i + 2)] if i < len(y) - 1 else E0
            Ptt1 = ptp[-(i + 1)]
            J = Ptt @ Phii.T @ np.linalg.pinv(Ptt1)

            xtt = xf[-(i + 2)] if i < len(y) - 1 else x0
            xtt1 = xp[-(i + 1)]
            xnt = xtt + J @ (xs[-1] - xtt1)
            Pnt = Ptt + J @ (pts[-1] - Ptt1) @ J.T
            
            ptt1s.append(pts[-1] @ J.T)
            xs.append(xnt)
            pts.append(Pnt)

        # Remember the lists are currently backwards, so
        # x0 and E0 are at the end
        *xs, x0 = xs
        *pts, E0 = pts

        xs = np.array(list(reversed(xs)))
        pt = np.array(list(reversed(pts)))
        ptt1s = np.array(list(reversed(ptt1s)))

        return xs, pt, ptt1s, x0, E0

    def log_likelihood(self, y, state_exog=None, obs_exog=None, params=None):
        if type(state_exog) == type(None):
            state_exog = self.default_state_exog(y)
        if type(obs_exog) == type(None):
            obs_exog = self.default_obs_exog(y)

        params = self.params() if params == None else params
        Phi, A, C, Q, R, U, x0, E0 = params
        xp, ptp, xf, ptf, ks = self.filter(y)
        ya = (U @ obs_exog.T).T

        At = A.T if A.ndim == 2 else np.transpose(A, axes=[0, 2, 1])
        return util.log_multivariate_normal(y, A @ xp + ya, A @ ptp @ At + R).sum()

    def expect(self, y, state_exog=None, obs_exog=None):
        if type(state_exog) == type(None):
            state_exog = self.default_state_exog(y)
        if type(obs_exog) == type(None):
            obs_exog = self.default_obs_exog(y)

        Phi, A, C, Q, R, U, x0, E0 = self.params()
        x, pt, ptt1, x0_, E0_ = self.smooth(y)

        xx = np.concatenate((np.expand_dims(x0_, [0]), x[:-1]), axis=0)
        ptpt = np.concatenate((np.expand_dims(E0_, [0]), pt[:-1]), axis=0)

        s11 = (x @ np.transpose(x, axes=[0, 2, 1]) + pt)
        s10 = (x @ np.transpose(xx, axes=[0, 2, 1]) + ptt1[:-1])
        s00 = (xx @ np.transpose(xx, axes=[0, 2, 1]) + ptpt)

        xa = C @ state_exog
        ya = U @ obs_exog
        
        if A.ndim == 2:
            e = y - (A @ x) - ya
        else:
            e = y - (A @ x) - ya

        return x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_
        
    log_levels = ['info', 'debug']
    log_level = 'info'

    def set_log_level(self, level):
        self.log_level = level

    def log(self, *args, level='info'):
        curIdx = self.log_levels.index(self.log_level)
        logIdx = self.log_levels.index(level)

        if logIdx <= curIdx:
            print(*args)

    def debug(self, *args):
        return self.log(*args, level='debug')

    def em_iter(self, y, state_exog=None, obs_exog=None, em_vars=['Phi', 'Q', 'A', 'C', 'R', 'U', 'x0', 'E0'], strict=False):
        if type(state_exog) == type(None):
            state_exog = self.default_state_exog(y)
        if type(obs_exog) == type(None):
            obs_exog = self.default_obs_exog(y)

        Phi, A, C, Q, R, U, x0, E0 = self.params()
        i = 0
        result = None

        def expect():
            nonlocal i, result
            
            if i == 0 or strict == True:
                result = self.expect(y, state_exog, obs_exog)

            i += 1
            return result

        prev_ll = None
        def debug_ll(param):
            nonlocal prev_ll
            # Do the if check here just to avoid computing the log_likelihood if we don't have to
            if self.log_level == 'debug':
                ll = self.log_likelihood(y, state_exog=state_exog, obs_exog=obs_exog)
                self.debug('{}: {}'.format(param, ll))
                
                if prev_ll != None and prev_ll > ll + 1e-4:
                    raise ValueError('{} likelihood decreased: {} -> {}'.format(param, prev_ll, ll))
                prev_ll = ll

        if 'Phi' in em_vars:
            if Phi.ndim > 2:
                raise ValueError('Cannot optimize Phi when it is initialized to be time-varying')

            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()
            xt = np.transpose(x, axes=[0, 2, 1])
            xxt = np.transpose(xx, axes=[0, 2, 1])

            if 'Phi' in self.constraints:
                Qi = np.linalg.inv(Q)
                f, D = self.constraints['Phi']
                qk = np.kron(Qi, s00)
                Phi1 = (Qi @ s10).reshape((s10.shape[0], Q.shape[1] ** 2))
                Phi1 -= qk @ f
                Phi1 -= (Qi @ xa @ xxt).reshape(Phi1.shape)
                Phi1 = Phi1 @ D
                Phi2 = D.T @ qk @ D
                phi = np.linalg.inv(Phi2.sum(axis=0)) @ Phi1.sum(axis=0)
                phi = f + D @ phi
                self.Phi = Phi = phi.reshape(Phi.shape)
            else:
                s00inv = util.safe_inverse(s00.sum(axis=0))
                self.Phi = Phi = (s10 - xa @ xxt).sum(axis=0) @ s00inv

            debug_ll('Phi')

        if 'Q' in em_vars:
            if Q.ndim > 2:
                raise ValueError('Cannot optimize Q when it is initialized to be time-varying')

            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()
            
            s10t = np.transpose(s10, axes=[0, 2, 1])
            xat = np.transpose(xa, axes=[0, 2, 1])
            xt = np.transpose(x, axes=[0, 2, 1])
            xxt = np.transpose(xx, axes=[0, 2, 1])

            Q = (
                s11 
                    - s10 @ Phi.T 
                    - Phi @ s10t 
                    - x @ xat
                    - xa @ xt 
                    + Phi @ s00 @ Phi.T
                    + Phi @ xx @ xat 
                    + xa @ xxt @ Phi.T 
                    + xa @ xat
            )
            
            if 'Q' in self.constraints:
                f, D = self.constraints['Q']
                q = Q.reshape((Q.shape[0], Q.shape[1] ** 2))
                q = np.linalg.inv(Q.shape[0] * D.T @ D) @ (q @ D).sum(axis=0)
                q = f + D @ q
                self.Q = Q = q.reshape(Q.shape[1:])
            else:
                self.Q = Q = Q.mean(axis=0)

            debug_ll('Q')

        if 'C' in em_vars:
            if C.ndim > 2:
                raise ValueError('Cannot optimize C when it is initialized to be time-varying')

            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()

            # The constraints are used to fit the exogenous coefficient matrix, this is why
            # the user can't specify custom constraints here
            f = np.zeros((C.shape[0] * C.shape[1], 1))
            D = np.kron(np.eye(C.shape[0]), np.transpose(state_exog, axes=[0, 2, 1]))
            Dt = np.transpose(D, axes=[0, 2, 1])
            xxt = np.transpose(xx, axes=[0, 2, 1])

            Qi = np.linalg.inv(Q)
            C1 = np.linalg.inv((Dt @ Qi @ D).sum(axis=0))
            C2 = (Dt @ Qi @ (x - np.kron(np.eye(Q.shape[0]), xxt) @ np.expand_dims(Phi.flatten(), [1]) - f)).sum(axis=0)
            c = C1 @ C2
            self.C = C = c.reshape(C.shape)
            # self.C = C = (x - Phi @ xx).mean(axis=0)
            # print(self.log_likelihood(y))
            # xx[0] = x0
            # self.C = C = (x - Phi @ xx).mean(axis=0)
            debug_ll('C')

        if 'x0' in em_vars:
            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()
            self.x0 = x0 = x0_
            debug_ll('x0')

        if 'E0' in em_vars:
            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()
            self.E0 = E0 = E0_
            debug_ll('E0')

        if 'A' in em_vars:
            if A.ndim > 2:
                raise ValueError('Cannot optimize A when it is initialized to be time-varying')

            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()
            xt = np.transpose(x, axes=[0, 2, 1])

            if 'A' in self.constraints:
                Ri = np.linalg.inv(R)

                f, D = self.constraints['A']

                # This is the reverse of the argument order used in the paper, but
                # passing them the other way doesn't seem to work.
                rk = np.kron(Ri, pt + x @ xt)
                
                A1 = Ri @ (y @ xt)
                A1 = A1.reshape((A1.shape[0], R.shape[0] * Q.shape[0]))
                A1 -= rk @ f
                A1 -= (Ri @ ya @ xt).reshape(A1.shape)
                A1 = A1 @ D

                A2 = D.T @ rk @ D
                a = np.linalg.inv(A2.sum(axis=0)) @ A1.sum(axis=0)
                self.A = A = (f + D @ a).reshape(A.shape)
            else:
                A1 = (y @ xt - ya @ xt).sum(axis=0)
                A2 = np.linalg.inv((pt + x @ xt).sum(axis=0))
                self.A = A = A1 @ A2

            debug_ll('A')

        if 'U' in em_vars:
            if U.ndim > 2:
                raise ValueError('Cannot optimize U when it is initialized to be time-varying')

            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()
            f = np.zeros((U.shape[0] * U.shape[1], 1))
            D = np.kron(np.eye(U.shape[0]), np.transpose(obs_exog, axes=[0, 2, 1]))
            Dt = np.transpose(D, axes=[0, 2, 1])
            
            Ri = np.linalg.inv(R)
            U1 = np.linalg.inv((Dt @ Ri @ D).sum(axis=0))
            U2 = (Dt @ Ri @ (y - A @ x - f)).sum(axis=0)
            u = U1 @ U2
            self.U = U = u.reshape(U.shape)

            debug_ll('U')

        if 'R' in em_vars:
            if R.ndim > 2:
                raise ValueError('Cannot optimize R when it is initialized to be time-varying')

            x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = expect()
            resid = (e @ np.transpose(e, axes=[0, 2, 1]))
            
            if A.ndim == 2:
                va = (A @ pt @ A.T)
            else:
                va = (A @ pt @ np.transpose(A, axes=[0, 2, 1]))

            R = (resid + va)
    
            if 'R' in self.constraints:
                f, D = self.constraints['R']
                r = R.reshape((R.shape[0], R.shape[1] ** 2))
                r = np.linalg.inv(R.shape[0] * D.T @ D) @ (r @ D).sum(axis=0)
                r = f + D @ r
                self.R = R = r.reshape(R.shape[1:])
            else:
                self.R = R = R.mean(axis=0)

            debug_ll('R')

    def em(self, y, state_exog=None, obs_exog=None, n=10, **kwargs):
        print('\t[0] ll: {:.2f}'.format(self.log_likelihood(y, state_exog=state_exog, obs_exog=obs_exog)))
        if type(state_exog) == type(None):
            state_exog = self.default_state_exog(y)
        if type(obs_exog) == type(None):
            obs_exog = self.default_obs_exog(y)

        prev = -np.inf
        tol = 1e-2

        for i in range(n):
            self.em_iter(y, state_exog, obs_exog, **kwargs)
            if (i + 1) % 10 == 0:
                ll = self.log_likelihood(y, state_exog=state_exog, obs_exog=obs_exog)
                print('\t[{}] ll: {:.2f}'.format(i + 1, ll))
                if ll < prev - tol:
                    raise ValueError('Error likelihood decreased ({} -> {})'.format(prev, ll))
                prev = ll
                if np.isnan(ll):
                    raise ValueError('nan encountered')

        # If we ended on a multiple o 10, the last ll was already printed
        if n % 10 != 0:
            ll = self.log_likelihood(y, state_exog=state_exog, obs_exog=obs_exog)
            print('[{}] ll: {:.2f}'.format(n, ll))

    def innovations(self, y, **kwargs):
        x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = self.expect(y, **kwargs)
        return e

    def standardized_innovations(self, y, **kwargs):
        x, xx, pt, ptt1, s11, s10, s00, e, xa, ya, x0_, E0_ = self.expect(y, **kwargs)
        Phi, A, C, Q, R, U, x0, E0 = self.params()
        At = A.T if A.ndim == 2 else np.transpose(A, axes=[0, 2, 1])
        E = A @ pt @ At + R
        Esq = util.sqrtm(E)
        return np.linalg.inv(Esq) @ e, Esq

    def innovations_to_data(self, e, Esq, Ks, params, state_exog=None, obs_exog=None):
        if type(state_exog) == type(None):
            state_exog = self.default_state_exog(e)
        if type(obs_exog) == type(None):
            obs_exog = self.default_obs_exog(e)

        # Note, we are using the passed in parameters here, not our own. These parameters
        # may be a subset of ours
        Phi, A, C, Q, R, U, x0, E0 = params
        ys = []
        xtt = x0
        xa = C @ state_exog
        ya = U @ obs_exog

        for i in range(e.shape[0]):
            v, ptsq, K = e[i], Esq[i], Ks[i]
            Phii = Phi if Phi.ndim == 2 else Phi[i]
            Ai = A if A.ndim == 2 else A[i]

            # Scale up the innovation
            v = ptsq @ v
            xtt1 = Phii @ xtt
            y = Ai @ xtt1 + ya[i] + v

            # Generate the next filtered state
            xtt = xtt1 + xa[i] + K @ v
            ys.append(y)

        return np.array(ys)

    def bootstrap(self, y, bs_vars, n=50, size=None, replace=True, **kwargs):
        params = self.params()
        e, Esq = self.standardized_innovations(y, **kwargs)
        xp, ptp, xf, ptf, ks = self.filter(y, **kwargs)
        
        if size == None:
            size = e.shape[0]
        
        if size < e.shape[0]:
            # If we only want to use a subset of the innovations,
            # then take a random sub-sample, and make sure
            # to update the parameters if we hav time-varying
            # parameters
            ix = np.random.choices(idxs, size=size, replace=False)
            e = e[ix]
            psub = []
            
            for p in params:
                psub.append(p[ix] if p.ndim == 3 else p)

            params = psub

        idxs = list(range(size))
        kfs = []

        for i in range(n):
            ix = np.random.choice(idxs, size=size, replace=replace)
            psub = []
            for p in params:
                psub.append(p[ix] if p.ndim == 3 else p)

            s = e[ix]
            Eq = Esq[ix]
            K = ks[ix]
            ysub = self.innovations_to_data(s, Eq, K, psub, **kwargs)
            Phi, A, C, Q, R, U, x0, E0 = psub
            kf = KalmanFilter(Phi, A, Q, R, x0=x0, E0=E0, C=C, U=U, constraints=self.constraints)
            kf.minimize(ysub, bs_vars)
            kfs.append(kf)

        print('Bootstrap results:')
        bsp = [k.named_params() for k in kfs]
        for v in bs_vars:
            p = np.array([b[v] for b in bsp])
            print('\t{}: {} ({} std)'.format(v, p.mean(axis=0), p.std(axis=0)))

        return kfs

    def minimize(self, y, optimize, state_exog=None, obs_exog=None, method='BFGS'):
        if type(state_exog) == type(None):
            state_exog = self.default_state_exog(y)
        if type(obs_exog) == type(None):
            obs_exog = self.default_obs_exog(y)

        p = self.named_params()
        args = []
        # The order is important here, because this is the order in which
        # they will be unpacked
        names = ['Phi', 'A', 'C', 'Q', 'R', 'U', 'x0', 'E0']
        constants = {}

        for name in names:
            if name in optimize:
                if name in self.constraints:
                    f, D = self.constraints[name]
                    if D.shape[1] == 0:
                        print('Warning: cannot optimize {} because its been constrained to have zero degrees of freedom, treating as constant'.format(name))
                        constants[name] = p[name]
                    else:
                        if name == 'Q' or name == 'R':
                            param = util.inverse_constrained_cholesky(f, D, np.linalg.cholesky(p[name]))
                        else:
                            param = np.linalg.pinv(D) @ (p[name].flatten() - f)
    
                        args.append(param)
                else:
                    if name == 'Q' or name == 'R':
                        args.append(util.cholesky_flatten(p[name]))
                    else:
                        args.append(p[name].flatten())
            else:
                constants[name] = p[name]

        i = 0
        def kl(*args, **kwargs):
            l = kalman_likelihood(*args, **kwargs)
            
            nonlocal i
            i += 1            
            if i % 100 == 0:
                print('[{}] ll: {:.2f}'.format(i, -l))

            return l

        params = np.concatenate(args)

        print('Starting likelihood: {:.2f} ({:.2f})'.format(
            -kalman_likelihood(params, constants, self.Q.shape[0], y, state_exog, obs_exog, self.constraints),
            self.log_likelihood(y, state_exog=state_exog, obs_exog=obs_exog)
        ))

        r = sp.optimize.minimize(
            kl,
            params,
            (
                constants,
                self.Q.shape[0],
                y,
                state_exog,
                obs_exog,
                self.constraints
            ),
            method=method
        )

        Phi, A, C, Q, R, U, x0, E0 = unflatten_kalman_params(
            r.x, 
            self.Q.shape[0],
            state_exog.shape[1],
            y.shape[1], 
            obs_exog.shape[1],
            constants,
            self.constraints
        )

        self.Phi = Phi
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.U = U
        self.x0 = x0
        self.E0 = E0
        print('Minimized: {:.2f}'.format(self.log_likelihood(y, state_exog=state_exog, obs_exog=obs_exog)))

def extract_parameter(i, params, shape, constraint, cholesky=False):
    if constraint != None:
        f, D = constraint
        P = params[i:i+D.shape[1]]
        i += D.shape[1]
        
        if cholesky == True:
            L = util.constrained_cholesky(f, D, P)
            P = L @ L.T
        else:
            P = f + D @ P
            P = P.reshape(shape)
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

def unflatten_kalman_params(params, state_dim, state_exog_dim, obs_dim, obs_exog_dim, args, constraints):
    i = 0

    if 'Phi' in args:
        Phi = args['Phi']
    else:
        Phi, i = extract_parameter(i, params, (state_dim, state_dim), constraints['Phi'] if 'Phi' in constraints else None)

    if 'A' in args:
        A = args['A']
    else:
        A, i = extract_parameter(i, params, (obs_dim, state_dim), constraints['A'] if 'A' in constraints else None)

    if 'C' in args:
        C = args['C']
    else:
        C, i = extract_parameter(i, params, (state_dim, state_exog_dim), constraints['C'] if 'C' in constraints else None)

    if 'Q' in args:
        Q = args['Q']
    else:
        Q, i = extract_parameter(i, params, (state_dim, state_dim), constraints['Q'] if 'Q' in constraints else None, cholesky=True)

    if 'R' in args:
        R = args['R']
    else:
        R, i = extract_parameter(i, params, (obs_dim, obs_dim), constraints['R'] if 'R' in constraints else None, cholesky=True)
   
    if 'U' in args:
        U = args['U']
    else:
        U, i = extract_parameter(i, params, (obs_dim, obs_exog_dim), constraints['U'] if 'U' in constraints else None)

    if 'x0' in args:
        x0 = args['x0']
    else:
        x0, i = extract_parameter(i, params, (state_dim, 1), None)

    if 'E0' in args:
        E0 = args['E0']
    else:
        E0, i = extract_parameter(i, params, (state_dim, state_dim), None, cholesky=True)
        
    return Phi, A, C, Q, R, U, x0, E0
    
def kalman_likelihood(params, args, state_dim, y, state_exog, obs_exog, constraints):
    Phi, A, C, Q, R, U, x0, E0 = unflatten_kalman_params(
        params, 
        state_dim,
        state_exog.shape[1],
        y.shape[1],
        obs_exog.shape[1],
        args,
        constraints
    )
    
    kf = KalmanFilter(
        Phi,
        A,
        Q,
        R,
        x0,
        E0,
        C,
        U
    )
    
    return -kf.log_likelihood(y, state_exog=state_exog, obs_exog=obs_exog)