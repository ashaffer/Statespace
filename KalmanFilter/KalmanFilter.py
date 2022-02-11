import numpy as np
import scipy as sp
from . import util
import re

class KalmanFilter:
    default_values = {
        'Phi': np.array([1.0]),
        'C': np.array([0.0]),
        'Q': np.array([0.05]),
        'A': np.array([1.0]),
        'U': np.array([0.0]),
        'R': np.array([0.05]),
        'x0': np.array([0.00]),
        'E0': np.array([0.05]),
    }

    def __init__(self, y, Phi=None, A=None, Q=None, R=None, x0=None, E0=None, C=None, U=None, G=None, H=None, F=None, inits={}, state_exog=None, obs_exog=None):
        obs_dim = y.shape[1]
        
        if Phi is not None:
            state_dim = Phi.shape[1]
        elif Q is not None:
            if G is not None:
                state_dim = G.shape[0] if G.ndim == 2 else G.shape[1]
            else:
                state_dim = Q.shape[1]
        elif x0 is not None:
            state_dim = x0.shape[0]
        elif E0 is not None:
            if F is not None:
                state_dim = F.shape[0]
            else:
                state_dim = E0.shape[0]
        elif C is not None:
            state_dim = C.shape[0] if C.ndim == 2 else C.shape[1]

        if state_exog is None:
            state_exog = self.default_state_exog(y)
        
        if obs_exog is None:
            obs_exog = self.default_obs_exog(y)

        for k in self.default_values:
            inits[k] = inits[k] if k in inits else self.default_values[k]

        if y.ndim == 1:
            y = y[:,None]
        if y.ndim == 2:
            y = y[:,:,None]

        if x0 is not None and x0.ndim == 1:
            x0 = x0[:,None]

        self.y = y
        self.state_exog = state_exog
        self.obs_exog = obs_exog
        self.constraints = {}
        self.variable_names = {}
        self.Phi = Phi = np.zeros((state_dim, state_dim)) if Phi is None else self.init_param('Phi', Phi, inits['Phi'])
        self.A = A = np.ones((obs_dim, state_dim)) if A is None else self.init_param('A', A, inits['A'])
        self.Q = Q = np.eye(state_dim) if Q is None else self.init_param('Q', Q, inits['Q'])
        self.R = R = np.eye(obs_dim) if R is None else self.init_param('R', R, inits['R'])
        self.G = G = np.eye(Q.shape[0]) if G is None else G
        self.H = H = np.eye(R.shape[0]) if H is None else H
        self.x0 = x0 = np.zeros((G.shape[0], 1)) if x0 is None else self.init_param('x0', x0, inits['x0'])
        self.E0 = E0 = np.zeros((Q.shape[-1], Q.shape[-1])) if E0 is None else self.init_param('E0', E0, inits['E0'])
        self.F = F = np.eye(E0.shape[-1]) if F is None else F        
        self.C = C = np.zeros((state_dim, state_exog.shape[1])) if C is None else self.init_param('C', C, inits['C'])
        self.U = U = np.zeros((obs_dim, obs_exog.shape[1])) if U is None else self.init_param('U', U, inits['U'])

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

        Qf = G @ Q @ G.T
        Rf = H @ R @ H.T
        E0f = F @ E0 @ F.T
        
        qz = np.diag(Qf) == 0
        rz = np.diag(Rf) == 0
        e0z = np.diag(E0f) == 0

        self.FF = np.linalg.inv(F.T @ F) @ F.T
        self.GG = np.linalg.inv(G.T @ G) @ G.T
        self.HH = np.linalg.inv(H.T @ H) @ H.T

        for key in self.constraints:
            C = self.constraints[key]

            if type(C) == str:
                C = util.constraint_str_to_matrix(C, params[key].shape)

            if type(C) == np.ndarray:
                self.constraints[key] = util.constraint_matrices(C)

        #
        # Generate stochasticity matrices for the degenerate variance
        # case (Section 7-8)
        #
        
        ds, inds, S = util.reachable_edges(util.to_adjacency(self.Phi), (~qz).astype(float))
        self.DS = DS = util.extend_matrices(self.y.shape[0] + 1, [np.eye(x0.shape[0]), *ds])
        self.IS = IS = util.extend_matrices(self.y.shape[0] + 1, [np.zeros(DS[0].shape), *inds])
        self.SS = SS = np.eye(state_dim) - DS

        # I_lambda in the paper (deterministic initial states)
        self.DE = DE = np.diagflat(e0z.astype(float))
        # I_l in the paper (stochastic initial states)
        self.SE = SE = np.eye(state_dim) - DE

        fc, Dc = self.constraint_idx('C', 0)
        fcstar = [np.zeros(fc.shape)]
        Dcstar = [np.zeros(Dc.shape)]

        for i in range(y.shape[0]):
            Phii, *_ = self.params_idx(i)
            fc, Dc = self.constraint_idx('C', i)
            fcstar.append(Phii @ fcstar[-1] + fc)
            Dcstar.append(Phii @ Dcstar[-1] + Dc)

        self.fcstar = np.array(fcstar)
        self.Dcstar = np.array(Dcstar)

    def describe_fit(self):
        print('Fitted:')
        for name in self.constraints:
            f, D = self.constraints[name]
            vnames = self.variable_names[name]
            M = getattr(self, name)
            p = np.linalg.pinv(D) @ (M.flatten()[:,None] - f)
            for i, v in enumerate(p[0]):
                print('\t{}: {:.4f}'.format(vnames[i], v[0]))

    def init_param(self, name, P, init):
        if P.ndim == 2:
            P = P[None]

        var_list = []
        isfloat = re.compile('^[\d+](?:\.\d+)?$')

        for t in range(P.shape[0]):
            for j in range(P.shape[1]):
                for k in range(P.shape[2]):
                    v = P[t][j][k]
                    if type(v) == np.str_:
                        p = re.split('[\+\-\*]', v)
                        p = [x.strip() for x in p if x != '']
                        for c in p:
                            if not isfloat.fullmatch(c):
                                if c not in var_list:
                                    var_list.append(c)

        vmap = {v: i for i,v in enumerate(var_list)}

        ndim = P.shape[1] * P.shape[2]
        f = np.zeros((P.shape[0], ndim, 1))
        D = np.zeros((P.shape[0], ndim, len(var_list)))

        for t in range(P.shape[0]):
            for j in range(P.shape[1]):
                for k in range(P.shape[2]):
                    v = P[t][j][k]
                    c = j * P.shape[2] + k

                    if util.is_number_type(v):
                        f[t][c] = float(v)
                    elif type(v) == np.str_:
                        p = re.split('([\+\-\*])', v)
                        p = [x.strip() for x in p if x != '']
                        sign = 1.0

                        while len(p):
                            if isfloat.fullmatch(p[0]):
                                if len(p) == 1:
                                    f[t][c] += sign * float(p[0])
                                    break
                                elif p[1] == '+' or p[1] == '-':
                                    f[t][c] += sign * float(p[0])
                                    sign = 1.0 if p[1] == '+' else -1.0
                                    p = p[2:]
                                elif p[1] == '*':
                                    if p[2] not in vmap:
                                        raise ValueError('Unrecognized variable: {}'.format(p[2]))

                                    D[t][c][vmap[p[2]]] = sign * float(p[0])
                                    sign = 1.0
                                    p = p[3:]
                                else:
                                    raise ValueError('Invalid variable string: {}'.format(v))
                            elif p[0] in vmap:
                                if len(p) == 1:
                                    D[t][c][vmap[p[0]]] = 1.0
                                    break
                                elif p[1] == '+' or p[1] == '-':
                                    sign = 1.0 if p[1] == '+' else -1.0
                                    D[t][c][vmap[p[0]]] = 1.0
                                    p = p[2:]
                                elif p[1] == '*' and isfloat.fullmatch(p[2]):
                                    D[t][c][vmap[p[0]]] = float(p[2]) * sign
                                    sign = 1.0
                                    p = p[3:]
                                else:
                                    raise ValueError('Invalid variable string: {}'.format(v))
                    else:
                        raise ValueError('Unrecognized type in {} ({})'.format(name, type(v)))

        self.variable_names[name] = var_list
        # If there are no variables, then it's just a fixed matrix
        # with no optimization constraints
        if D.shape[-1] == 0:
            if P.shape[0] == 1:
                return P[0]
            else:
                return P

        if name in ['Phi', 'Q', 'R', 'E0']:
            if init.shape != P.shape[-2:]:
                if init.shape[0] == 1:
                    M = np.diagflat(np.full(P.shape[-1], init[0]))
                elif init.shape == P.shape[-1] and P.shape[-2] == P.shape[-1]:
                    M = np.diagflat(init)
                else:
                    raise ValueError('Unrecognized init format')

            Dm = D.mean(0)
            fm = f.mean(0)
            v = np.linalg.pinv(Dm) @ (M.flatten()[:,None] - fm)
        else:
            if len(init) == 1:
                v = np.full((len(vmap), 1), init)
            elif len(init) == len(vmap):
                v = init
            else:
                raise ValueError('Mis-specified init vector for: {}'.format(name))

        P = (f + D @ v).reshape(P.shape)
        if P.shape[0] == 1:
            P = P[0]
        
        self.constraints[name] = (f, D)
        return P

    def is_state_degenerate(self):
        return (np.abs(self.G).sum(1) == 0).any()

    def is_obs_degenerate(self):
        return (np.abs(self.H).sum(1) == 0).any()

    def is_degenerate(self):
        return self.is_state_degenerate() or self.is_obs_degenerate()

    def default_state_exog(self, y):
        return np.ones((len(y), 1, 1))
    
    def default_obs_exog(self, y):
        return np.ones((len(y), 1, 1))

    def named_params(self):
        return {
            'Phi': self.Phi,
            'A': self.A,
            'C': self.C,
            'Q': self.Q,
            'R': self.R,
            'U': self.U,
            'G': self.G,
            'H': self.H,
            'x0': self.x0,
            'E0': self.E0,
            'F': self.F
        }

    def constraint_idx(self, name, i, **kwargs):
        if name not in self.constraints:
            P = getattr(self, name)
            ndim = P.shape[-2] * P.shape[-1]
            f = np.zeros((ndim, 1))
            D = np.eye(ndim)
            return f, D

        f, D = self.constraints[name]        
        f = f[i] if f.shape[0] > 1 else f[0]
        D = D[i] if D.shape[0] > 1 else D[0]

        return f, D

    def params_idx(self, i, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F = self.endog_params(**kwargs)

        return (
            Phi if Phi.ndim == 2 else Phi[i], 
            A if A.ndim == 2 else A[i], 
            C, 
            Q if Q.ndim == 2 else Q[i], 
            R if R.ndim == 2 else R[i], 
            U, 
            x0,
            E0,
            G if G.ndim == 2 else G[i],
            H if H.ndim == 2 else H[i],
            F
        )

    def endog_params(self, **kwargs):
        Phi = kwargs['Phi'] if 'Phi' in kwargs else self.Phi
        A = kwargs['A'] if 'A' in kwargs else self.A
        C = kwargs['C'] if 'C' in kwargs else self.C 
        Q = kwargs['Q'] if 'Q' in kwargs else self.Q
        R = kwargs['R'] if 'R' in kwargs else self.R
        U = kwargs['U'] if 'U' in kwargs else self.U
        x0 = kwargs['x0'] if 'x0' in kwargs else self.x0
        E0 = kwargs['E0'] if 'E0' in kwargs else self.E0
        G = kwargs['G'] if 'G' in kwargs else self.G
        H = kwargs['H'] if 'H' in kwargs else self.H
        F = kwargs['F'] if 'F' in kwargs else self.F

        return Phi, A, C, Q, R, U, x0, E0, G, H, F
    
    def params(self, **kwargs):
        return *self.endog_params(**kwargs), self.state_exog, self.obs_exog

    def filter_once(self, i, xtt1, Ptt1, y, ya=0, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F = self.params_idx(i, **kwargs)

        resid = y - A @ xtt1 - ya
        K = Ptt1 @ A.T @ util.symm(np.linalg.pinv(A @ Ptt1 @ A.T + H @ R @ H.T))
        xtt = xtt1 + K @ resid
        Ptt = (np.eye(Ptt1.shape[0]) - K @ A) @ Ptt1
        Ptt = util.symm(Ptt)

        return xtt, Ptt, K

    def predict_once(self, i, xtt, Ptt, xa=0, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F = self.params_idx(i, **kwargs)
        xtt1 = Phi @ xtt + xa
        Ptt1 = Phi @ Ptt @ Phi.T + G @ Q @ G.T
        Ptt1 = util.symm(Ptt1)

        return xtt1, Ptt1

    def likelihood_once(self, i, xtt1, Ptt1, y, ya=0, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F = self.params_idx(i, **kwargs)

        return util.multivariate_normal_density(
            y,
            A @ xtt1 + ya,
            A @ Ptt1 @ A.T + H @ R @ H.T
        )

    def filter(self, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()

        xp = []
        xf = []
        ptf = []
        ptp = []
        ks = []

        # Pre-compute the state and obs intercepts for the whole series
        xa = self.state_intercept()
        ya = self.obs_intercept()
        xtt = x0 
        Ptt = F @ E0 @ F.T

        for i,v in enumerate(self.y):
            xtt1, Ptt1 = self.predict_once(i, xtt, Ptt, xa[i])
            xtt, Ptt, K = self.filter_once(i, xtt1, Ptt1, v, ya[i])
            if np.isnan(xtt).any():
                print(i)

            xp.append(xtt1)
            xf.append(xtt)
            ptp.append(Ptt1)
            ptf.append(Ptt)
            ks.append(K)

        return (
            np.array(xp),
            np.array(ptp),
            np.array(xf),
            np.array(ptf),
            np.array(ks)
        )

    def smooth_cov_init(self, K, Pn1, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F = self.params_idx(-1, **kwargs)
        I = np.eye(K.shape[0])
        return (I - K @ A) @ Phi @ Pn1

    def smooth_once(self, i, xs, Pts, xtt, xtt1, Ptt, Pt1t1, prev_cov, xa=0, ya=0, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F = self.params_idx(i, **kwargs)

        Ptt1 = Phi @ Ptt @ Phi.T + G @ Q @ G.T
        J = Ptt @ Phi.T @ util.symm(util.safe_inverse(Ptt1))

        xnt = xtt + J @ (xs - xtt1)
        Pnt = Ptt + J @ (Pts - Ptt1) @ J.T
        Pnt = util.symm(Pnt)

        if Pt1t1 is not None:
            Pt1t = util.symm(Phi @ Pt1t1 @ Phi.T + G @ Q @ G.T)
            J2 = Pt1t1 @ Phi.T @ util.symm(util.safe_inverse(Pt1t))
            cov = Ptt @ J2.T + J @ (prev_cov - Phi @ Ptt) @ J2.T
        else:
            cov = Pts @ J.T

        return xnt, Pnt, cov

    def smooth(self):
        filter_params = self.filter()
        return self._smooth(filter_params)

    def _smooth(self, filter_params, cov_init=None, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        xp, ptp, xf, ptf, ks = filter_params
        y = self.y

        xs = [xf[-1]]
        pts = [ptf[-1]]
        covs = [
            self.smooth_cov_init(ks[-1], ptf[-2]) if cov_init is None else cov_init
        ]

        E0f = F @ E0 @ F.T

        for i in range(1, len(xp)):
            xnt, Pnt, cov = self.smooth_once(
                -i,
                xs[-1], 
                pts[-1],
                xf[-(i + 1)], 
                xp[-i], 
                ptf[-(i + 1)],
                ptf[-(i + 2)] if i < len(xp) - 1 else E0f,
                covs[-1],
                **kwargs
            )

            xs.append(xnt)
            pts.append(Pnt)
            covs.append(cov)

        x0, E0f, cov0 = self.smooth_once(0, xs[-1], pts[-1], x0, xp[0], E0f, None, covs[-1], **kwargs)
        covs.append(cov0)

        E0 = self.FF @ E0f @ self.FF.T

        return (
            np.array(list(reversed(xs))),
            np.array(list(reversed(pts))),
            np.array(list(reversed(covs))),
            x0,
            E0
        )


    def _log_likelihoods(self, filter_params, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params(**kwargs)
        xp, ptp, xf, ptf, ks = filter_params
        ya = self.obs_intercept()
        At = A.T if A.ndim == 2 else np.transpose(A, axes=[0, 2, 1])

        mu = (A @ xp) + ya
        s2 = A @ ptp @ At + H @ R @ H.T

        if np.isnan(mu).any() or np.isnan(s2).any():
            return np.nan

        return util.log_multivariate_normal(self.y, mu, s2)

    def log_likelihoods(self, **kwargs):
        filter_params = self.filter()
        return self._log_likelihoods(filter_params, **kwargs)

    def log_likelihood(self, **kwargs):
        return self.log_likelihoods(**kwargs).sum()

    def em_params(self, **kwargs):
        smooth_params = self.smooth(**kwargs)
        return self._em_params(smooth_params)

    def _em_params(self, smooth_params, **kwargs):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params(**kwargs)
        x, pt, covs, x0_, E0_ = smooth_params

        # Only update the stochastic elements of x0
        x0_ = self.SE @ x0_ + self.DE @ x0

        xx = np.concatenate((np.expand_dims(x0_, [0]), x[:-1]), axis=0)
        ptpt = np.concatenate((np.expand_dims(F @ E0_ @ F.T, [0]), pt[:-1]), axis=0)

        xt = np.transpose(x, axes=[0, 2, 1])
        xxt = np.transpose(xx, axes=[0, 2, 1])

        # Element-wise outer products of the x's
        s11 = x @ xt + pt
        s10 = x @ xxt + covs[1:]
        s00 = xx @ xxt + ptpt

        xa = self.state_intercept()
        ya = self.obs_intercept()

        e = self.y - (A @ x) - ya

        At = A.T if A.ndim == 2 else np.transpose(A, axes=[0, 2, 1])
        obs_cov = A @ pt @ At
        resid_cov = e @ np.transpose(e, axes=[0, 2, 1]) + obs_cov

        return x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov

    def em_Phi(self, em_params, constraints={}):
        if self.is_degenerate():
            raise ValueError('EM updates for the Phi parameter in degenerate models are not currently supported')

        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params
        xxt = np.transpose(xx, axes=[0, 2, 1])

        if Phi.ndim > 2:
            raise ValueError('Cannot optimize Phi when it is initialized to be time-varying')

        if self.is_degenerate() > 0:
            raise ValueError('Optimizing Phi unimplemented for degenerate variance models')

        if 'Phi' in constraints:
            GG = self.GG
            Qi = GG.T @ util.safe_inverse(Q) @ GG
            f, D = constraints['Phi']
            qk = np.kron(Qi, s00)
            Phi1 = (Qi @ (s10 - (C @ xxt))).reshape((s10.shape[0], x0.shape[0] ** 2, 1))

            Dt = np.transpose(D, axes=[0, 2, 1])
            Phi1 -= qk @ f
            Phi1 = Dt @ Phi1
            Phi2 = Dt @ qk @ D
            phi = util.safe_inverse(Phi2.sum(0)) @ Phi1.sum(0)
            phi = f + D @ phi
            Phi = phi.reshape((D.shape[0], *Phi.shape[-2:]))
            if Phi.shape[0] == 1:
                Phi = Phi[0]
        else:
            s00inv = util.safe_inverse(s00.sum(0))
            num = (s10 - xa @ xxt).sum(0)
            Phi = num @ s00inv

        return Phi

    def em_Q(self, em_params, constraints={}):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params

        if Q.ndim > 2:
            raise ValueError('Cannot optimize Q when it is initialized to be time-varying')

        s10t = np.transpose(s10, axes=[0, 2, 1])
        xt = np.transpose(x, axes=[0, 2, 1])
        xxt = np.transpose(xx, axes=[0, 2, 1])        
        xat = np.transpose(xa, axes=[0, 2, 1])

        qs = (
            s11 
                - s10 @ Phi.T 
                - Phi @ s10t
                - x @ C.T
                - C @ xt
                + Phi @ s00 @ Phi.T
                + Phi @ xx @ C.T
                + C @ xxt @ Phi.T 
                + C @ C.T
        )

        qs = self.GG @ qs @ self.GG.T
        self.qs = qs
        if 'Q' in constraints:
            f, D = constraints['Q']
            Dt = np.transpose(D, axes=[0, 2, 1])

            q = qs.reshape((qs.shape[0], qs.shape[1] ** 2, 1))
            self.q = q            
            q = util.safe_inverse(qs.shape[0] * Dt @ D) @ (Dt @ q).sum(0)
            q = f + D @ q
            Q = q.reshape((D.shape[0], *qs.shape[-2:]))
            if Q.shape[0] == 1:
                Q = Q[0]
        else:
            self.qs = qs
            Q = qs.mean(0)

        Q = util.symm(Q)
        return Q

    def em_C(self, em_params, constraints={}):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params

        if C.ndim > 2:
            raise ValueError('Cannot optimize C when it is initialized to be time-varying')

        if 'C' in constraints:
            f, D = constraints['C']
        else:
            Csz = x0.shape[0] * state_exog.shape[1]
            f = np.zeros((1, Csz, 1))
            D = np.eye(Csz)[None]

        Qf = G @ Q @ G.T
        Rf = H @ R @ H.T

        Phistar = [np.eye(x0.shape[0])]
        
        for i in range(self.y.shape[0]):
            Phii, *_ = self.params_idx(i)
            Phistar.append(Phii @ Phistar[-1])

        self.Phistar = np.array(Phistar)

        fcstar, Dcstar, DE, SE, DS, SS = self.fcstar, self.Dcstar, self.DE, self.SE, self.DS, self.SS

        Delta1 = self.y - A @ SS[1:] @ x - (A @ DS[1:] @ (Phistar[1:] @ x0 + fcstar[1:])) - ya
        Delta2 = A @ DS[1:] @ Dcstar[1:]
        Delta3 = x - Phi @ SS[:-1] @ xx - Phi @ DS[:-1] @ (Phistar[:-1] @ x0 + fcstar[:-1]) - f
        Delta3[0] = x[0] - Phi @ x0 - f[0]
        Delta4 = D + Phi @ DS[:-1] @ Dcstar[:-1]
        Delta4[0] = D

        GG = self.GG
        Qi = GG.T @ util.safe_inverse(Q) @ GG

        HH = self.HH
        Ri = HH.T @ util.safe_inverse(R) @ HH

        Delta4t = np.transpose(Delta4, axes=[0, 2, 1])
        Delta2t = np.transpose(Delta2, axes=[0, 2, 1])

        V1 = Delta4t @ Qi @ Delta3 + Delta2t @ Ri @ Delta1
        V2 = Delta4t @ Qi @ Delta4 + Delta2t @ Ri @ Delta2
        v = np.linalg.inv(V2.sum(0)) @ V1.sum(0)

        c = f + D @ v
        C = c.reshape((D.shape[0], *C.shape[-2:]))
        if C.shape[0] == 1:
            C = C[0]

        return C

    def em_x0(self, em_params, constraints={}):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params

        V0f = F @ E0 @ F.T
        z = np.diag(V0f) == 0.0
        if z.all():
            if 'x0' in self.constraints:
                f, D = self.constraints['x0']
                DS, SS, DE, SE = self.DS, self.SS, self.DE, self.SE
        
                Cstar = [0 * (C[0] if C.ndim == 3 else C)]
                Phistar = [np.eye(x0.shape[0])]

                for i in range(self.y.shape[0]):
                    Phii, Ai, Ci, Qi, Ri, Ui, x0, E0, Gi, Hi, Fi = self.params_idx(i)
                    Phistar.append(Phii @ Phistar[-1])
                    Cstar.append(Phii @ Cstar[-1] + Ci)

                Phistar = np.array(Phistar)
                Cstar = np.array(Cstar)

                Delta5 = self.y - A @ SS[1:] @ x - A @ DS[1:] @ ((Phistar[1:] @ SE @ x0 + DE @ f) + Cstar[1:]) - U
                Delta6 = A @ DS[1:] @ Phistar[1:] @ DE @ D
                Delta7 = x - Phi @ SS[:-1] @ xx - Phi @ DS[:-1] @ ((Phistar[:-1] @ SE @ x0 + DE @ f) + Cstar[:-1]) - C
                Delta7[0] = x[0] - Phi @ (SE @ x0 + DE @ f[0]) - (C[0] if C.ndim == 3 else C)
                Delta8 = Phi @ DS[:-1] @ Phistar[:-1] @ DE @ D
                Delta8[0] = Phi @ DE @ D[0]

                Dt = np.transpose(D, axes=[0, 2, 1])
                Delta6t = np.transpose(Delta6, axes=[0, 2, 1])
                Delta8t = np.transpose(Delta8, axes=[0, 2, 1])

                GG = self.GG
                Qi = GG.T @ util.safe_inverse(Q) @ GG

                HH = self.HH
                Ri = HH.T @ util.safe_inverse(R) @ HH

                FF = np.linalg.inv(F.T @ F) @ F.T
                E0i = FF.T @ util.safe_inverse(E0) @ FF

                V1 = Delta8t @ Qi @ Delta8 + Delta6t @ Ri @ Delta6 + Dt @ E0i @ D
                V2 = Delta8t @ Qi @ Delta7 + Delta6t @ Ri @ Delta5 + Dt @ E0i @ (x0 - f)

                p = np.linalg.inv(V1.sum(0)) @ V2.sum(0)
                x0_ = f + D @ p
                x0_ = x0_.reshape(x0.shape)
            else:
                GG = self.GG
                Qi = GG.T @ util.safe_inverse(Q) @ GG
                x0_ = np.linalg.inv(Phi.T @ Qi @ Phi) @ Phi.T @ Qi @ (x[0] - xa[0])
        elif z.any():
            raise ValueError('Partial stochasticity of x0 is not currently supported')
            # Im = np.eye(x0.shape[0])
            # TT5 = y - A @ S[1:] @ x - A @ DS[1:] @ (Bstar @ ((Im - self.IE) @ x0 + self.IE @ f) - ya)

        return x0_

    def em_E0(self, em_params, constraints={}):
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params
        return E0_

    def em_A(self, em_params, constraints={}):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params

        if A.ndim > 2:
            raise ValueError('Cannot optimize A when it is initialized to be time-varying')

        if 'A' in constraints:
            HH = self.HH
            Ri = HH.T @ util.safe_inverse(R) @ HH

            f, D = constraints['A']
            Dt = np.transpose(D, axes=[0, 2, 1])
            xt = np.transpose(x, axes=[0, 2, 1])

            # This is the reverse of the argument order used in the paper, but
            # passing them the other way doesn't seem to work.
            rk = np.kron(Ri, pt + x @ xt)
            
            A1 = Ri @ (self.y @ xt)
            A1 = A1.reshape((A1.shape[0], H.shape[0] * G.shape[0], 1))
            A1 -= rk @ f
            A1 -= (Ri @ ya @ xt).reshape(A1.shape)
            A1 = Dt @ A1

            A2 = Dt @ rk @ D
            a = np.linalg.inv(A2.sum(axis=0)) @ A1.sum(axis=0)
            A = (f + D @ a).reshape((D.shape[0], *A.shape[-2:]))
            if A.shape[0] == 1:
                A = A[0]
        else:
            xt = np.transpose(x, axes=[0, 2, 1])
            A1 = (self.y @ xt - ya @ xt).sum(axis=0)
            A2 = np.linalg.inv((pt + x @ xt).sum(axis=0))
            A = A1 @ A2

        return A

    def em_U(self, em_params, constraints={}):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params

        if U.ndim > 2:
            raise ValueError('Cannot optimize U when it is initialized to be time-varying')

        if 'U' in constraints:
            f, D = constraints['U']
        else:
            Usz = H.shape[0] * obs_exog.shape[1]
            f = np.zeros((Usz, 1))
            D = np.eye(Usz)

        # Note: the identity matrix here can be replaced with a time varying
        # input matrix if we want, but that is currently unimplemented.
        Ddt = np.kron(np.transpose(state_exog, axes=[0, 2, 1]), np.eye(H.shape[0]))
        Dddt = Ddt @ D
        Dddtt = np.transpose(Dddt, axes=[0, 2, 1])
        ftd = Dddt @ f
        
        HH = self.HH
        Ri = HH.T @ util.safe_inverse(R) @ HH

        U1 = np.linalg.inv((Dddtt @ Ri @ Dddt).sum(0))
        U2 = (Dddtt @ Ri @ (e + ya)).sum(0)

        u = U1 @ U2
        U = u.reshape((D.shape[0], *U.shape[-2:]))
        if U.shape[0] == 1:
            U = U[0]

        return U

    def em_R(self, em_params, constraints={}):
        Phi, A, C, Q, R, U, x0, E0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = em_params

        if R.ndim > 2:
            raise ValueError('Cannot optimize R when it is initialized to be time-varying')

        resid_cov = self.HH @ resid_cov @ self.HH.T

        if 'R' in constraints:
            f, D = constraints['R']
            Dt = np.transpose(D, axes=[0, 2, 1])
            r = resid_cov.reshape((resid_cov.shape[0], resid_cov.shape[1] ** 2, 1))
            r = np.linalg.inv(resid_cov.shape[0] * Dt @ D) @ (Dt @ r).sum(axis=0)
            r = f + D @ r
            R = r.reshape((D.shape[0], *resid_cov.shape[1:]))
            if R.shape[0] == 1:
                R = R[0]
        else:
            R = resid_cov.mean(axis=0)

        R = util.symm(R)
        return R

        
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

    def em_once(self, em_vars=['Phi', 'Q', 'A', 'C', 'R', 'U', 'x0', 'E0'], starting_likelihood=None, strict=False, constraints={}, i=0):
        # Order shouldn't matter if we're operating in strict mode where we re-compute the filters/smoothers
        # on every iteration, but that's not the default, and I think order might matter in non-strict mode
        # order = ['Phi', 'Q', 'A', 'C', 'R', 'U', 'x0', 'E0']    
        order = ['R', 'Q', 'x0', 'E0', 'U', 'C', 'Phi', 'A']
        prev_ll = starting_likelihood
        em_params = None

        for name in order:
            if name in em_vars:
                mname = 'em_{}'.format(name)
                if not hasattr(self, mname):
                    raise ValueError('[KalmanFilter] No EM maximizer available for: {}'.format(name))

                if em_params is None or strict == True:
                    em_params = self.em_params()

                fn = getattr(self, mname)
                result = fn(em_params, constraints=constraints)
                setattr(self, name, result)

                # Do the if check here just to avoid computing the log_likelihood if we don't have to
                if self.log_level == 'debug':
                    ll = self.log_likelihood()
                    print('\t{}: {}'.format(name, ll))
                    
                    if prev_ll != None and prev_ll > ll + 1e-4:
                        print('[KalmanFilter] iter {}: {} likelihood decreased: {} -> {}'.format(i, name, prev_ll, ll))
                        # raise ValueError('{} likelihood decreased: {} -> {}'.format(name, prev_ll, ll))
                    prev_ll = ll

        return prev_ll

    def em(self, n=10, **kwargs):
        if 'em_vars' not in kwargs or kwargs['em_vars'] is None:
            kwargs['em_vars'] = list(self.constraints.keys())

        ll = self.log_likelihood()
        print('\t[0] ll: {:.6f}'.format(ll))

        prev = -np.inf
        tol = 1e-2

        for i in range(n):
            ll = self.em_once(starting_likelihood=ll, constraints=self.constraints, i=i, **kwargs)
            if (i + 1) % 10 == 0:
                ll = self.log_likelihood()
                print('\t[{}] ll: {:.6f}'.format(i + 1, ll))
                if ll < prev - tol:
                    raise ValueError('Error likelihood decreased ({} -> {})'.format(prev, ll))
                prev = ll
                if np.isnan(ll):
                    raise ValueError('nan encountered')

        # If we ended on a multiple o 10, the last ll was already printed
        if n % 10 != 0:
            ll = self.log_likelihood()
            print('[{}] ll: {:.6f}'.format(n, ll))

    def innovations(self, **kwargs):
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = self.em_params(**kwargs)
        return e

    def standardized_innovations(self, **kwargs):
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, E0_, resid_cov = self.em_params(**kwargs)
        Phi, A, C, Q, R, U, x0, E0, G, H, F = self.params()
        At = A.T if A.ndim == 2 else np.transpose(A, axes=[0, 2, 1])
        E = A @ pt @ At + R
        Esq = util.sqrtm(E)
        return np.linalg.inv(Esq) @ e, Esq

    def state_intercept(self):
        return (self.C @ self.state_exog.T).T

    def obs_intercept(self):
        return (self.U @ self.obs_exog.T).T

    def innovations_to_data(self, e, Esq, Ks, params, state_exog=None, obs_exog=None):
        # Note, we are using the passed in parameters here, not our own. These parameters
        # may be a subset of ours
        Phi, A, C, Q, R, U, x0, E0 = params
        ys = []
        xtt = x0
        xa = self.state_intercept(e, state_exog)
        ya = self.obs_intercept(e, obs_exog)

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


    def minimize(self, optimize=None, method='BFGS'):
        args = []
        constants = {}
        p = self.named_params()

        if optimize is None:
            optimize = list(self.constraints.keys())

        for name in p:
            if name in optimize:
                if name in self.constraints:
                    f, D = self.constraints[name]
                    if D.shape[-1] == 0:
                        print('Warning: cannot optimize {} because its been constrained to have zero degrees of freedom, treating as constant'.format(name))
                        constants[name] = p[name]
                    else:
                        if name == 'Q' or name == 'R' or name == 'E0':
                            if D.shape[0] != 1 or f.shape[0] != 1:
                                raise ValueError('Time-varying {} is not allowed when optimizing'.format(name))

                            param = util.inverse_constrained_cholesky(f[0], D[0], util.safe_cholesky(p[name]))[:,None]
                        else:
                            D = D.mean(0)
                            f = f.mean(0)
                            param = np.linalg.pinv(D) @ (p[name].flatten()[:,None] - f)

                        args.append(param)
                else:
                    if name == 'Q' or name == 'R':
                        args.append(util.cholesky_flatten(p[name]))
                    else:
                        args.append(p[name].flatten()[:,None])
            else:
                constants[name] = p[name]

        i = 0

        def kl(*args, **kwargs):
            l = self.params_likelihood(*args, **kwargs)
            
            nonlocal i
            i += 1            
            if i % 100 == 0:
                print('[{}] ll: {:.2f}'.format(i, -l))

            return l

        params = np.concatenate(args)

        print('Starting likelihood: {:.2f} ({:.2f})'.format(
            -self.params_likelihood(params, constants),
            self.log_likelihood()
        ))

        r = sp.optimize.minimize(
            kl,
            params,
            (constants, ),
            method=method
        )

        params = self.unflatten_params(
            r.x, 
            # self.x0.shape[0],
            # state_exog.shape[1],
            # y.shape[1], 
            # obs_exog.shape[1],
            constants,
            self.constraints
        )

        for p in params:
            setattr(self, p, params[p])

        print('Minimized: {:.2f}'.format(self.log_likelihood()))
        self.describe_fit()

    cholesky_params = ['Q', 'R', 'E0']

    def unflatten_params(self, buf, args, constraints, i=0):
        params = {}
        
        for p in self.named_params():
            if p in args:
                params[p] = args[p]
            else:
                P = getattr(self, p)
                v, i = self.extract_parameter(i, buf, P.shape, constraints[p] if p in constraints else None, cholesky=p in self.cholesky_params)
                params[p] = v

        return params

    # @classmethod
    # def unflatten_params(cls, params, state_dim, state_exog_dim, obs_dim, obs_exog_dim, args, constraints, i=0):
    #     if 'Phi' in args:
    #         Phi = args['Phi']
    #     else:
    #         Phi, i = cls.extract_parameter(i, params, (state_dim, state_dim), constraints['Phi'] if 'Phi' in constraints else None)

    #     if 'A' in args:
    #         A = args['A']
    #     else:
    #         A, i = cls.extract_parameter(i, params, (obs_dim, state_dim), constraints['A'] if 'A' in constraints else None)

    #     if 'C' in args:
    #         C = args['C']
    #     else:
    #         C, i = cls.extract_parameter(i, params, (state_dim, state_exog_dim), constraints['C'] if 'C' in constraints else None)

    #     if 'Q' in args:
    #         Q = args['Q']
    #     else:
    #         Q, i = cls.extract_parameter(i, params, (state_dim, state_dim), constraints['Q'] if 'Q' in constraints else None, cholesky=True)

    #     if 'R' in args:
    #         R = args['R']
    #     else:
    #         R, i = cls.extract_parameter(i, params, (obs_dim, obs_dim), constraints['R'] if 'R' in constraints else None, cholesky=True)
       
    #     if 'U' in args:
    #         U = args['U']
    #     else:
    #         U, i = cls.extract_parameter(i, params, (obs_dim, obs_exog_dim), constraints['U'] if 'U' in constraints else None)

    #     if 'x0' in args:
    #         x0 = args['x0']
    #     else:
    #         x0, i = cls.extract_parameter(i, params, (state_dim, 1), None)

    #     if 'E0' in args:
    #         E0 = args['E0']
    #     else:
    #         E0, i = cls.extract_parameter(i, params, (state_dim, state_dim), None, cholesky=True)
            
    #     return Phi, A, C, Q, R, U, x0, E0

    @staticmethod
    def extract_parameter(i, params, shape, constraint, cholesky=False):
        if params.ndim == 1:
            params = params[:,None]

        if constraint != None:
            f, D = constraint
            P = params[i:i+D.shape[-1]]
            i += D.shape[-1]
            
            if cholesky == True:
                L = util.constrained_cholesky(f[0], D[0], P)
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
            n = np.prod(shape)
            P = params[i:i+n]
            P = P.reshape(shape)
            i += n

        return P, i

    def params_likelihood(self, params, args):
        params = self.unflatten_params(
            params,
            args,
            self.constraints
        )

        kf = self.__class__(self.y, state_exog=self.state_exog, obs_exog=self.obs_exog, **params)    
        return -kf.log_likelihood()