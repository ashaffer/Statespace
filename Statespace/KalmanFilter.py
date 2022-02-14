import numpy as np
import scipy as sp
from . import util
import re

class KalmanFilter:
    default_values = {
        'B': np.array([1.0]),
        'U': np.array([0.0]),
        'Q': np.array([0.05]),
        'G': np.array([1.0]),
        'Z': np.array([1.0]),
        'A': np.array([0.0]),
        'R': np.array([0.05]),
        'H': np.array([1.0]),
        'x0': np.array([0.00]),
        'V0': np.array([0.05]),
    }

    def __init__(self, y, B=None, Z=None, Q=None, R=None, x0=None, V0=None, U=None, A=None, G=None, H=None, F=None, inits={}, state_exog=None, obs_exog=None):
        # Expand the dimensions to 3. i.e. n_samples, m_obs, 1
        if y.ndim == 1:
            y = y[:,None]
        if y.ndim == 2:
            y = y[:,:,None]

        if x0 is not None and x0.ndim == 1:
            x0 = x0[:,None]

        # Do our best to infer all the dimensions from whatever is specified, so that we can
        # generate default values for the unspecified ones.
        obs_dim = y.shape[1]
        
        if B is not None:
            state_dim = B.shape[1]
        elif Q is not None:
            if G is not None:
                state_dim = G.shape[0] if G.ndim == 2 else G.shape[1]
            else:
                state_dim = Q.shape[1]
        elif x0 is not None:
            state_dim = x0.shape[0]
        elif V0 is not None:
            if F is not None:
                state_dim = F.shape[0]
            else:
                state_dim = V0.shape[0]
        elif U is not None:
            state_dim = U.shape[0] if U.ndim == 2 else U.shape[1]

        # By default we include a constant term, in order to 
        # fit an intercept if nothing else is specified
        if state_exog is None:
            state_exog = self.default_state_exog(y)
        
        if obs_exog is None:
            obs_exog = self.default_obs_exog(y)

        # Iniitialize any unspecified parameters to reasonable defaults
        for k in self.default_values:
            inits[k] = inits[k] if k in inits else self.default_values[k]

        self.y = y
        self.state_exog = state_exog
        self.obs_exog = obs_exog
        self.constraints = {}
        self.variable_names = {}
        self.B = B = np.zeros((state_dim, state_dim)) if B is None else self.init_param('B', B, inits['B'])
        self.Z = Z = np.ones((obs_dim, state_dim)) if Z is None else self.init_param('Z', Z, inits['Z'])
        self.Q = Q = np.eye(state_dim) if Q is None else self.init_param('Q', Q, inits['Q'])
        self.R = R = np.eye(obs_dim) if R is None else self.init_param('R', R, inits['R'])
        self.G = G = np.eye(Q.shape[0]) if G is None else self.init_param('G', G, inits['G'])
        self.H = H = np.eye(R.shape[0]) if H is None else self.init_param('H', H, inits['H'])
        self.x0 = x0 = np.zeros((G.shape[0], 1)) if x0 is None else self.init_param('x0', x0, inits['x0'])
        self.V0 = V0 = np.zeros((state_dim, state_dim)) if V0 is None else self.init_param('V0', V0, inits['V0'])
        self.F = F = np.eye(V0.shape[-1]) if F is None else F        
        self.U = U = np.zeros((state_dim, state_exog.shape[1])) if U is None else self.init_param('U', U, inits['U'])
        self.A = A = np.zeros((obs_dim, obs_exog.shape[1])) if A is None else self.init_param('A', A, inits['A'])

        self.validate_dims()

        # Pre-generate matrices that are useful for extracting the degenerate
        # states/observations from the likelihoods. It's ok to pre-generate
        # these things because the degenerate components are required
        # to be fixed over time.
        Qf = G @ Q @ G.T
        Rf = H @ R @ H.T
        V0f = F @ V0 @ F.T
        
        qz = np.diag(Qf) == 0
        rz = np.diag(Rf) == 0
        e0z = np.diag(V0f) == 0

        self.FF = np.linalg.inv(F.T @ F) @ F.T
        self.GG = np.linalg.inv(G.T @ G) @ G.T
        self.HH = np.linalg.inv(H.T @ H) @ H.T

        params = {
            'B': self.B,
            'Z': self.Z,
            'Q': self.Q,
            'R': self.R,
            'x0': self.x0,
            'V0': self.V0,
            'U': self.U,
            'A': self.A
        }

        for key in self.constraints:
            U = self.constraints[key]

            if type(U) == str:
                U = util.constraint_str_to_matrix(U, params[key].shape)

            if type(U) == np.ndarray:
                self.constraints[key] = util.constraint_matrices(U)

        #
        # Generate stochasticity matrices for the degenerate variance
        # case (Section 7-8)
        #
        
        ds, inds, S = util.reachable_edges(util.to_adjacency(self.B), (~qz).astype(float))
        self.DS = DS = util.extend_matrices(self.y.shape[0] + 1, [np.eye(x0.shape[0]), *ds])
        self.IS = IS = util.extend_matrices(self.y.shape[0] + 1, [np.zeros(DS[0].shape), *inds])
        self.SS = SS = np.eye(state_dim) - DS

        # I_lambda in the paper (deterministic initial states)
        self.DE = DE = np.diagflat(e0z.astype(float))
        # I_l in the paper (stochastic initial states)
        self.SE = SE = np.eye(state_dim) - DE

        # These are only using the f/D matrix structures over time
        # *not* the actual values of the U parameter, which is why
        # its ok to pre-generate them like this.
        fu, Du = self.constraint_idx('U', 0)
        fustar = [np.zeros(fu.shape)]
        Dustar = [np.zeros(Du.shape)]

        for i in range(y.shape[0]):
            Bi, *_ = self.params_idx(i)
            fu, Du = self.constraint_idx('U', i)
            fustar.append(Bi @ fustar[-1] + fu)
            Dustar.append(Bi @ Dustar[-1] + Du)

        self.fustar = np.array(fustar)
        self.Dustar = np.array(Dustar)

    def validate_dims(self):
        n = self.y[0]
        x0_cov_dim = self.F.shape[-1]
        state_cov_dim = self.Q.shape[-1]
        state_dim = self.G.shape[-2]
        state_exog_dim = self.state_exog.shape[-1]
        obs_cov_dim = self.H.shape[-1]
        obs_dim = self.y.shape[1]
        obs_exog_dim = self.obs_exog.shape[-1]

        def validate(name, dim, exp):
            if dim != exp and dim != (n, *exp):
                print('Parameter: {} dimensions are invalid'.format(name))
                print('\t{} expected'.format(exp))
                print('\t{} actual'.format(dim))
                raise ValueError('Invalid parameters')

        validate('x0', self.x0.shape, (state_dim, 1))
        validate('V0', self.V0.shape, (x0_cov_dim, x0_cov_dim))
        validate('F', self.F.shape, (state_dim, x0_cov_dim))
        validate('B', self.B.shape, (state_dim, state_dim))
        validate('U', self.U.shape, (state_dim, state_exog_dim))
        validate('G', self.G.shape, (state_dim, state_cov_dim))
        validate('Q', self.Q.shape, (state_cov_dim, state_cov_dim))
        validate('Z', self.Z.shape, (obs_dim, state_dim))
        validate('A', self.A.shape, (obs_dim, obs_exog_dim))
        validate('H', self.H.shape, (obs_dim, obs_cov_dim))
        validate('R', self.R.shape, (obs_cov_dim, obs_cov_dim))

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

        if type(init) == list or type(init) == tuple:
            init = np.array(init)

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

        if name in ['B', 'Q', 'R', 'V0']:
            if init.shape != P.shape[-2:]:
                if init.shape[0] == 1:
                    M = np.diagflat(np.full(P.shape[-1], init[0]))
                elif init.shape == P.shape[-1] and P.shape[-2] == P.shape[-1]:
                    M = np.diagflat(init)
                else:
                    raise ValueError('Unrecognized init format')
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

        if v.ndim == 1:
            v = v[:,None]

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
            'B': self.B,
            'Z': self.Z,
            'U': self.U,
            'Q': self.Q,
            'R': self.R,
            'A': self.A,
            'G': self.G,
            'H': self.H,
            'x0': self.x0,
            'V0': self.V0,
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
        B, Z, U, Q, R, A, x0, V0, G, H, F = self.endog_params(**kwargs)

        return (
            B if B.ndim == 2 else B[i], 
            Z if Z.ndim == 2 else Z[i], 
            U, 
            Q if Q.ndim == 2 else Q[i], 
            R if R.ndim == 2 else R[i], 
            A, 
            x0,
            V0,
            G if G.ndim == 2 else G[i],
            H if H.ndim == 2 else H[i],
            F
        )

    def endog_params(self, **kwargs):
        B = kwargs['B'] if 'B' in kwargs else self.B
        Z = kwargs['Z'] if 'Z' in kwargs else self.Z
        U = kwargs['U'] if 'U' in kwargs else self.U 
        Q = kwargs['Q'] if 'Q' in kwargs else self.Q
        R = kwargs['R'] if 'R' in kwargs else self.R
        A = kwargs['A'] if 'A' in kwargs else self.A
        x0 = kwargs['x0'] if 'x0' in kwargs else self.x0
        V0 = kwargs['V0'] if 'V0' in kwargs else self.V0
        G = kwargs['G'] if 'G' in kwargs else self.G
        H = kwargs['H'] if 'H' in kwargs else self.H
        F = kwargs['F'] if 'F' in kwargs else self.F

        return B, Z, U, Q, R, A, x0, V0, G, H, F
    
    def params(self, **kwargs):
        return *self.endog_params(**kwargs), self.state_exog, self.obs_exog

    def filter_once(self, i, xtt1, Ptt1, y, ya=0, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F = self.params_idx(i, **kwargs)

        resid = y - Z @ xtt1 - ya
        K = Ptt1 @ Z.T @ util.symm(np.linalg.pinv(Z @ Ptt1 @ Z.T + H @ R @ H.T))
        xtt = xtt1 + K @ resid
        Ptt = (np.eye(Ptt1.shape[0]) - K @ Z) @ Ptt1
        Ptt = util.symm(Ptt)

        return xtt, Ptt, K

    def predict_once(self, i, xtt, Ptt, xa=0, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F = self.params_idx(i, **kwargs)
        xtt1 = B @ xtt + xa
        Ptt1 = B @ Ptt @ B.T + G @ Q @ G.T
        Ptt1 = util.symm(Ptt1)

        return xtt1, Ptt1

    def likelihood_once(self, i, xtt1, Ptt1, y, ya=0, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F = self.params_idx(i, **kwargs)

        return util.multivariate_normal_density(
            y,
            Z @ xtt1 + ya,
            Z @ Ptt1 @ Z.T + H @ R @ H.T
        )

    def filter(self, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()

        xp = []
        xf = []
        ptf = []
        ptp = []
        ks = []

        # Pre-compute the state and obs intercepts for the whole series
        xa = self.state_intercept()
        ya = self.obs_intercept()
        xtt = x0 
        Ptt = F @ V0 @ F.T

        for i,v in enumerate(self.y):
            xtt1, Ptt1 = self.predict_once(i, xtt, Ptt, xa[i])
            xtt, Ptt, K = self.filter_once(i, xtt1, Ptt1, v, ya[i])

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
        B, Z, U, Q, R, A, x0, V0, G, H, F = self.params_idx(-1, **kwargs)
        I = np.eye(K.shape[0])
        return (I - K @ Z) @ B @ Pn1

    def smooth_once(self, i, xs, Pts, xtt, xtt1, Ptt, Pt1t1, prev_cov, xa=0, ya=0, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F = self.params_idx(i, **kwargs)

        Ptt1 = B @ Ptt @ B.T + G @ Q @ G.T
        J = Ptt @ B.T @ util.symm(util.safe_inverse(Ptt1))

        xnt = xtt + J @ (xs - xtt1)
        Pnt = Ptt + J @ (Pts - Ptt1) @ J.T
        Pnt = util.symm(Pnt)

        if Pt1t1 is not None:
            Pt1t = util.symm(B @ Pt1t1 @ B.T + G @ Q @ G.T)
            J2 = Pt1t1 @ B.T @ util.symm(util.safe_inverse(Pt1t))
            cov = Ptt @ J2.T + J @ (prev_cov - B @ Ptt) @ J2.T
        else:
            cov = Pts @ J.T

        return xnt, Pnt, cov

    def smooth(self):
        filter_params = self.filter()
        return self._smooth(filter_params)

    def _smooth(self, filter_params, cov_init=None, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        xp, ptp, xf, ptf, ks = filter_params
        y = self.y

        xs = [xf[-1]]
        pts = [ptf[-1]]
        covs = [
            self.smooth_cov_init(ks[-1], ptf[-2]) if cov_init is None else cov_init
        ]

        V0f = F @ V0 @ F.T

        for i in range(1, len(xp)):
            xnt, Pnt, cov = self.smooth_once(
                -i,
                xs[-1], 
                pts[-1],
                xf[-(i + 1)], 
                xp[-i], 
                ptf[-(i + 1)],
                ptf[-(i + 2)] if i < len(xp) - 1 else V0f,
                covs[-1],
                **kwargs
            )

            xs.append(xnt)
            pts.append(Pnt)
            covs.append(cov)

        x0, V0f, cov0 = self.smooth_once(0, xs[-1], pts[-1], x0, xp[0], V0f, None, covs[-1], **kwargs)
        covs.append(cov0)

        V0 = self.FF @ V0f @ self.FF.T

        return (
            np.array(list(reversed(xs))),
            np.array(list(reversed(pts))),
            np.array(list(reversed(covs))),
            x0,
            V0
        )


    def _log_likelihoods(self, filter_params, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params(**kwargs)
        xp, ptp, xf, ptf, ks = filter_params
        ya = self.obs_intercept()
        Zt = Z.T if Z.ndim == 2 else np.transpose(Z, axes=[0, 2, 1])

        mu = (Z @ xp) + ya
        s2 = Z @ ptp @ Zt + H @ R @ H.T

        if np.isnan(mu).any() or np.isnan(s2).any():
            return np.nan

        return util.log_multivariate_normal(self.y, mu, s2)

    def log_likelihoods(self, **kwargs):
        filter_params = self.filter()
        return self._log_likelihoods(filter_params, **kwargs)

    def log_likelihood(self, **kwargs):
        lls = self.log_likelihoods(**kwargs)
        
        if np.isnan(lls).any():
            return np.nan

        return lls.sum()

    def em_params(self, **kwargs):
        smooth_params = self.smooth(**kwargs)
        return self._em_params(smooth_params)

    def _em_params(self, smooth_params, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params(**kwargs)
        x, pt, covs, x0_, V0_ = smooth_params

        # Only update the stochastic elements of x0
        x0_ = self.SE @ x0_ + self.DE @ x0

        xx = np.concatenate((np.expand_dims(x0_, [0]), x[:-1]), axis=0)
        ptpt = np.concatenate((np.expand_dims(F @ V0_ @ F.T, [0]), pt[:-1]), axis=0)

        xt = np.transpose(x, axes=[0, 2, 1])
        xxt = np.transpose(xx, axes=[0, 2, 1])

        # Element-wise outer products of the x's
        s11 = x @ xt + pt
        s10 = x @ xxt + covs[1:]
        s00 = xx @ xxt + ptpt

        xa = self.state_intercept()
        ya = self.obs_intercept()

        e = self.y - (Z @ x) - ya

        Zt = Z.T if Z.ndim == 2 else np.transpose(Z, axes=[0, 2, 1])
        obs_cov = Z @ pt @ Zt
        resid_cov = e @ np.transpose(e, axes=[0, 2, 1]) + obs_cov

        return x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov

    def em_B(self, em_params):
        if self.is_degenerate():
            raise ValueError('EM updates for the B parameter in degenerate models are not currently supported')

        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params
        xxt = np.transpose(xx, axes=[0, 2, 1])

        if B.ndim > 2:
            raise ValueError('Cannot optimize B when it is initialized to be time-varying')

        if self.is_degenerate() > 0:
            raise ValueError('Optimizing B unimplemented for degenerate variance models')

        if 'B' in self.constraints:
            GG = self.GG
            Qi = GG.T @ util.safe_inverse(Q) @ GG
            f, D = self.constraints['B']
            qk = np.kron(Qi, s00)
            B1 = (Qi @ (s10 - (U @ xxt))).reshape((s10.shape[0], x0.shape[0] ** 2, 1))

            Dt = np.transpose(D, axes=[0, 2, 1])
            B1 -= qk @ f
            B1 = Dt @ B1
            B2 = Dt @ qk @ D
            b = util.safe_inverse(B2.sum(0)) @ B1.sum(0)
            b = f + D @ b
            B = b.reshape((D.shape[0], *B.shape[-2:]))
            if B.shape[0] == 1:
                B = B[0]
        else:
            s00inv = util.safe_inverse(s00.sum(0))
            num = (s10 - xa @ xxt).sum(0)
            B = num @ s00inv

        return B

    def em_Q(self, em_params):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        if Q.ndim > 2:
            raise ValueError('Cannot optimize Q when it is initialized to be time-varying')

        s10t = np.transpose(s10, axes=[0, 2, 1])
        xt = np.transpose(x, axes=[0, 2, 1])
        xxt = np.transpose(xx, axes=[0, 2, 1])        
        xat = np.transpose(xa, axes=[0, 2, 1])

        qs = (
            s11 
                - s10 @ B.T 
                - B @ s10t
                - x @ U.T
                - U @ xt
                + B @ s00 @ B.T
                + B @ xx @ U.T
                + U @ xxt @ B.T 
                + U @ U.T
        )

        qs = self.GG @ qs @ self.GG.T
        self.qs = qs
        if 'Q' in self.constraints:
            f, D = self.constraints['Q']
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

    def _em_U_degenerate(self, em_params, f, D, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params(**kwargs)
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        Qf = G @ Q @ G.T
        Rf = H @ R @ H.T

        Bstar = [np.eye(x0.shape[0])]
        
        for i in range(self.y.shape[0]):
            Bi, *_ = self.params_idx(i, **kwargs)
            Bstar.append(Bi @ Bstar[-1])

        self.Bstar = np.array(Bstar)

        fustar, Dustar, DE, SE, DS, SS = self.fustar, self.Dustar, self.DE, self.SE, self.DS, self.SS

        Xt0 = self.expected_x0(em_params)
        Delta1 = self.y - Z @ SS[1:] @ x - (Z @ DS[1:] @ (Bstar[1:] @ Xt0 + fustar[1:])) - ya
        Delta2 = Z @ DS[1:] @ Dustar[1:]
        Delta3 = x - B @ SS[:-1] @ xx - B @ DS[:-1] @ (Bstar[:-1] @ Xt0 + fustar[:-1]) - f
        Delta3[0] = x[0] - B @ Xt0 - f[0]
        Delta4 = D + B @ DS[:-1] @ Dustar[:-1]
        Delta4[0] = D

        GG = self.GG
        Qi = GG.T @ util.safe_inverse(Q) @ GG

        HH = self.HH
        Ri = HH.T @ util.safe_inverse(R) @ HH

        Delta4t = np.transpose(Delta4, axes=[0, 2, 1])
        Delta2t = np.transpose(Delta2, axes=[0, 2, 1])

        num = Delta4t @ Qi @ Delta3 + Delta2t @ Ri @ Delta1
        denom = Delta4t @ Qi @ Delta4 + Delta2t @ Ri @ Delta2
        return num, denom

    def em_U(self, em_params):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        if U.ndim > 2:
            raise ValueError('Cannot optimize U when it is initialized to be time-varying')

        if 'U' in self.constraints:
            f, D = self.constraints['U']
        else:
            Usz = x0.shape[0] * state_exog.shape[1]
            f = np.zeros((1, Usz, 1))
            D = np.eye(Usz)[None]

        num, denom = self._em_U_degenerate(em_params, f, D)
        v = util.safe_inverse(denom.sum(0)) @ num.sum(0)

        u = f + D @ v
        U = u.reshape((D.shape[0], *U.shape[-2:]))
        if U.shape[0] == 1:
            U = U[0]

        return U

    def expected_x0(self, em_params, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params(**kwargs)        
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params
        # Take the stochastic components from the latest estimate, and the non-stochastic
        # components from the prior estimate
        SE, DE = self.SE, self.DE
        return SE @ x0_ + DE @ x0

    def _em_x0_degenerate(self, em_params, f, D, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params(**kwargs)
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        DS, SS, DE, SE = self.DS, self.SS, self.DE, self.SE
    
        Ustar = [0 * (U[0] if U.ndim == 3 else U)]
        Bstar = [np.eye(x0.shape[0])]

        for i in range(self.y.shape[0]):
            Bi, Zi, Ui, Qi, Ri, Ai, x0, V0, Gi, Hi, Fi = self.params_idx(i, **kwargs)
            Bstar.append(Bi @ Bstar[-1])
            Ustar.append(Bi @ Ustar[-1] + Ui)

        Bstar = np.array(Bstar)
        Ustar = np.array(Ustar)

        Delta5 = self.y - Z @ SS[1:] @ x - Z @ DS[1:] @ ((Bstar[1:] @ SE @ x0 + DE @ f) + Ustar[1:]) - A
        Delta6 = Z @ DS[1:] @ Bstar[1:] @ DE @ D
        Delta7 = x - B @ SS[:-1] @ xx - B @ DS[:-1] @ ((Bstar[:-1] @ SE @ x0 + DE @ f) + Ustar[:-1]) - U
        Delta7[0] = x[0] - B @ (SE @ x0 + DE @ f[0]) - (U[0] if U.ndim == 3 else U)
        Delta8 = B @ DS[:-1] @ Bstar[:-1] @ DE @ D
        Delta8[0] = B @ DE @ D[0]

        Dt = np.transpose(D, axes=[0, 2, 1])
        Delta6t = np.transpose(Delta6, axes=[0, 2, 1])
        Delta8t = np.transpose(Delta8, axes=[0, 2, 1])

        GG = self.GG
        Qi = GG.T @ util.safe_inverse(Q) @ GG

        HH = self.HH
        Ri = HH.T @ util.safe_inverse(R) @ HH

        FF = np.linalg.inv(F.T @ F) @ F.T
        V0i = FF.T @ util.safe_inverse(V0) @ FF

        Xt0 = self.expected_x0(em_params)

        num = Delta8t @ Qi @ Delta7 + Delta6t @ Ri @ Delta5 + Dt @ V0i @ (Xt0 - f)        
        denom = Delta8t @ Qi @ Delta8 + Delta6t @ Ri @ Delta6 + Dt @ V0i @ D

        return num, denom

    def em_x0(self, em_params):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        V0f = F @ V0 @ F.T
        z = np.diag(V0f) == 0.0
        if z.all():
            if 'x0' in self.constraints:
                f, D = self.constraints['x0']
                num, denom = self._em_x0_degenerate(em_params)
                p = util.safe_inverse(denom.sum(0)) @ num.sum(0)
                x0_ = f + D @ p
                x0_ = x0_.reshape(x0.shape)
            else:
                GG = self.GG
                Qi = GG.T @ util.safe_inverse(Q) @ GG
                x0_ = np.linalg.inv(B.T @ Qi @ B) @ B.T @ Qi @ (x[0] - xa[0])
        elif z.any():
            raise ValueError('Partial stochasticity of x0 is not currently supported')
            # Im = np.eye(x0.shape[0])
            # TT5 = y - Z @ S[1:] @ x - Z @ DS[1:] @ (Bstar @ ((Im - self.IE) @ x0 + self.IE @ f) - ya)

        return x0_

    def em_V0(self, em_params):
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params
        return V0_

    def em_Z(self, em_params):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        if Z.ndim > 2:
            raise ValueError('Cannot optimize Z when it is initialized to be time-varying')

        if 'Z' in self.constraints:
            HH = self.HH
            Ri = HH.T @ util.safe_inverse(R) @ HH

            f, D = self.constraints['Z']
            Dt = np.transpose(D, axes=[0, 2, 1])
            xt = np.transpose(x, axes=[0, 2, 1])

            # This is the reverse of the argument order used in the paper, but
            # passing them the other way doesn't seem to work.
            rk = np.kron(Ri, pt + x @ xt)
            
            Z1 = Ri @ (self.y @ xt)
            Z1 = Z1.reshape((Z1.shape[0], H.shape[0] * G.shape[0], 1))
            Z1 -= rk @ f
            Z1 -= (Ri @ ya @ xt).reshape(Z1.shape)
            Z1 = Dt @ Z1

            Z2 = Dt @ rk @ D
            a = np.linalg.inv(Z2.sum(axis=0)) @ Z1.sum(axis=0)
            Z = (f + D @ a).reshape((D.shape[0], *Z.shape[-2:]))
            if Z.shape[0] == 1:
                Z = Z[0]
        else:
            xt = np.transpose(x, axes=[0, 2, 1])
            Z1 = (self.y @ xt - ya @ xt).sum(axis=0)
            Z2 = np.linalg.inv((pt + x @ xt).sum(axis=0))
            Z = Z1 @ Z2

        return Z

    def em_A(self, em_params):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        if A.ndim > 2:
            raise ValueError('Cannot optimize A when it is initialized to be time-varying')

        if 'A' in self.constraints:
            f, D = self.constraints['A']
        else:
            Asz = H.shape[0] * obs_exog.shape[1]
            f = np.zeros((Asz, 1))
            D = np.eye(Asz)

        # Note: the identity matrix here can be replaced with a time varying
        # input matrix if we want, but that is currently unimplemented.
        Ddt = np.kron(np.transpose(state_exog, axes=[0, 2, 1]), np.eye(H.shape[0]))
        Dddt = Ddt @ D
        Dddtt = np.transpose(Dddt, axes=[0, 2, 1])
        ftd = Dddt @ f
        
        HH = self.HH
        Ri = HH.T @ util.safe_inverse(R) @ HH

        A1 = np.linalg.inv((Dddtt @ Ri @ Dddt).sum(0))
        A2 = (Dddtt @ Ri @ (e + ya)).sum(0)

        u = A1 @ A2
        A = u.reshape((D.shape[0], *A.shape[-2:]))
        if A.shape[0] == 1:
            A = A[0]

        return A

    def em_R(self, em_params):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params()
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, *_ = em_params

        if R.ndim > 2:
            raise ValueError('Cannot optimize R when it is initialized to be time-varying')

        resid_cov = self.HH @ resid_cov @ self.HH.T

        if 'R' in self.constraints:
            f, D = self.constraints['R']
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

    def fit(self, em_tol=0.1):
        self.em(tol=em_tol)
        self.minimize()

    def em_once(self, em_vars=['B', 'Q', 'Z', 'U', 'R', 'A', 'x0', 'V0'], starting_likelihood=None, strict=False, i=0):
        # Order shouldn't matter if we're operating in strict mode where we re-compute the filters/smoothers
        # on every iteration, but that's not the default, and I think order might matter in non-strict mode
        # order = ['B', 'Q', 'Z', 'U', 'R', 'A', 'x0', 'V0']    
        order = ['R', 'Q', 'x0', 'V0', 'A', 'U', 'B', 'Z']
        prev_ll = starting_likelihood
        em_params = None

        for name in em_vars:
            if name not in order:
                if name == 'G' or name == 'H':
                    print('Warning: No EM updater is implemented for {} yet. The EM pass will not fit your {} variables, you can fit them in a second pass with .minimize()'.format(name, name))
                else:
                    raise ValueError('[KalmanFilter] Unrecognized em_var: {}'.format(name))

        for name in order:
            if name in em_vars:
                mname = 'em_{}'.format(name)
                if not hasattr(self, mname):
                    raise ValueError('[KalmanFilter] No EM maximizer available for: {}'.format(name))

                if em_params is None or strict == True:
                    em_params = self.em_params()

                fn = getattr(self, mname)
                result = fn(em_params)
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

    def em(self, n=10, tol=None, eps=1e-2, em_vars=None, **kwargs):
        if em_vars is None:
            em_vars = list(self.constraints.keys())

        print('Starting EM on: {}'.format(', '.join(em_vars)))

        ll = self.log_likelihood()
        print('\t[0] ll: {:.6f}'.format(ll))

        prev = -np.inf

        if tol is not None:
            n = 10000

        for i in range(n):
            ll = self.em_once(starting_likelihood=ll, i=i, em_vars=em_vars, **kwargs)

            if (i + 1) % 10 == 0:
                ll = self.log_likelihood()

                print('\t[{}] ll: {:.6f}'.format(i + 1, ll))

                if np.isnan(ll):
                    raise ValueError('nan encountered')

                delta = ll - prev        

                if tol is not None and delta < tol:                    
                    break

                if delta < -eps:
                    raise ValueError('Error likelihood decreased ({} -> {})'.format(prev, ll))

                prev = ll

        # If we ended on a multiple o 10, the last ll was already printed
        if n % 10 != 0:
            ll = self.log_likelihood()
            print('[{}] ll: {:.6f}'.format(n, ll))

        self.describe_fit()
        print('')

    def innovations(self, **kwargs):
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov = self.em_params(**kwargs)
        return e

    def standardized_innovations(self, **kwargs):
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov = self.em_params(**kwargs)
        B, Z, U, Q, R, A, x0, V0, G, H, F = self.params()
        Zt = Z.T if Z.ndim == 2 else np.transpose(Z, axes=[0, 2, 1])
        E = Z @ pt @ Zt + R
        Esq = util.sqrtm(E)
        return np.linalg.inv(Esq) @ e, Esq

    def state_intercept(self):
        return (self.U @ self.state_exog.T).T

    def obs_intercept(self):
        return (self.A @ self.obs_exog.T).T

    def innovations_to_data(self, e, Esq, Ks, params, state_exog=None, obs_exog=None):
        # Note, we are using the passed in parameters here, not our own. These parameters
        # may be a subset of ours
        B, Z, U, Q, R, A, x0, V0 = params
        ys = []
        xtt = x0
        xa = self.state_intercept(e, state_exog)
        ya = self.obs_intercept(e, obs_exog)

        for i in range(e.shape[0]):
            v, ptsq, K = e[i], Esq[i], Ks[i]
            Bi = B if B.ndim == 2 else B[i]
            Zi = Z if Z.ndim == 2 else Z[i]

            # Scale up the innovation
            v = ptsq @ v
            xtt1 = Bi @ xtt
            y = Zi @ xtt1 + ya[i] + v

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
            B, Z, U, Q, R, A, x0, V0 = psub
            kf = KalmanFilter(B, Z, Q, R, x0=x0, V0=V0, U=U, A=A, constraints=self.constraints)
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
                        if name == 'Q' or name == 'R' or name == 'V0':
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
        print('Minimizing: {}'.format(', '.join(optimize)))
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

    cholesky_params = ['Q', 'R', 'V0']

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