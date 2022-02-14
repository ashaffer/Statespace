import numpy as np
import scipy as sp
from . import util
import scipy.stats
# from hmmlearn.hmm import GaussianHMM as hl_GaussianHMM
from .KalmanFilter import *

# def cloneHmmLearn(gh):
#     model = hl_GaussianHMM(n_components=gh.Phi.shape[0], covariance_type='diag')
#     model.startprob_ = gh.pi
#     model.transmat_ = gh.Phi
#     model.means_ = np.expand_dims(np.array([x.mean() for x in gh.dists]), [1])
#     model.covars_ = np.expand_dims(np.array([x.var() for x in gh.dists]), [1])
#     return model
#     

class HMM:
    def __init__(self, P, dists, pi=None):
        self.dists = dists
        # self.dists = [sp.stats.poisson(mu) for mu in state_params]
        # self.state_params = state_params
        self.P = P
        self.pi = util.solve_stationary(P.T) if type(pi) == type(None) else pi
        self.state_dim = self.P.shape[0]

    def hmm_params(self):
        return self.P, self.pi

    def filter_once(self, ptt1, py):
        P, pi = self.hmm_params()
        pt = (py * ptt1) / (py @ ptt1)
        ptt1 = P @ pt 
        return pt, ptt1

    def filter(self, y):
        # My textbook and other implementations seem to set ptt1 = self.pi rather than pt,
        # but the EM algorithm decreases in likelihood occasionally when I do it that way, and
        # this way it does not. It also seems more logical this way to me.
        pt = self.pi
        ptt1 = self.P @ pt
        pred = []
        filtered = []
        py = self.pdf(y)

        for i,v in enumerate(y):
            pred.append(ptt1)
            pt, ptt1 = self.filter_once(ptt1, py[i])
            filtered.append(pt)

        return np.array(pred), np.array(filtered), py

    def pdf(self, val, pt=None, state=None):
        result = []

        for d in self.dists:
            if isinstance(d.dist, sp.stats.rv_continuous):
                v = d.pdf(val)
                
                # Temporary hack, fix later
                if v.ndim == 1 and v.shape[0] == 1:
                    v = v[0]

                result.append(v)
            else:
                result.append(d.pmf(val))

        return np.array(result).T

    def smooth_once(self, p, pt, py):
        P, pi = self.hmm_params()

        # We only care about using the relative weights of p, and if we alllow it to recursively
        # multiply backwards, it just keeps shrinking in scale. So just periodically rescale it.
        if p.min() < 1e-15 or p.max() > 1e15:
            p = p / p.sum()

        prevp = p
        p = P @ (py * p)
        pnt = (pt * p) / (pt @ p)
        pxn = (pnt * (P * py * prevp).T / p).T

        return p, pnt, pxn

    def smooth_init_pxn(self, pnt):
        return (pnt * self.P.T).T

    def smooth(self, y):
        ptp, ptf, py = self.filter(y)
        states = [ptf[-1]]
        p = np.ones(self.pi.shape)
        pxns = [self.smooth_init_pxn(ptf[-1])]

        for i in range(1, len(y)):
            p, pnt, pxn = self.smooth_once(p, ptf[-(i+1)], py[-i])
            states.append(pnt)
            pxns.append(pxn)

        _, pn0, _ = self.smooth_once(p, self.pi, py[0])

        states = list(reversed(states))
        pxns = list(reversed(pxns))

        return np.array(states), np.array(pxns), pn0

    def sample(self, n):
        svals = np.random.uniform(0, 1, size=n)
        st = np.where(svals[0] <= self.pi.cumsum())[0][0]
        states = [st]
        vals = [self.dists[st].rvs(1)[0]]

        for sv in svals[1:]:
            st = np.where(sv <= self.P[st].cumsum())[0][0]
            states.append(st)
            vals.append(self.dists[st].rvs(1)[0])

        return np.array(states), np.array(vals)

    def log_likelihood(self, y):
        ptp, ptf, py = self.filter(y)
        dv = (ptp * py).sum(axis=1) # Probability weighted sum of the masses
        return np.log(dv).sum()

    def minimize(self, y, method='BFGS'):
        params = np.concatenate((
            self.pi,
            self.P.flatten(),
            *[d.params() for d in self.dists]
        ))

        i = 0
        def hmm_ll(params):
            nonlocal i

            ll = create_hmm(params, self.pi.size, self.__class__).log_likelihood(y)

            i += 1
            if i % 100 == 0:
                print('\t[{}] {:.2f}'.format(i, ll))

            return -ll

        print('Starting likelihood: {:.2f} ({:.2f})'.format(self.log_likelihood(y), hmm_ll(params)))
        r = sp.optimize.minimize(
            hmm_ll,
            params,
            method=method
        )

        hmm = create_hmm(r.x, self.pi.size, self.__class__)

        self.pi = hmm.pi
        self.P = hmm.P
        self.dists = hmm.dists
        print('Minimized: {:.2f}'.format(self.log_likelihood(y)))


    def em_distributions(self, st, pxns, y):
        # This is to be overriden by distributional specific child classes
        pass

    def em(self, y, n=10, strict=False):
        # Note: Estimating the U parameter can lead to instability, due to its feedback effects
        # on probability estimation. That is, the state intercept can influence what the HMM
        # believes the states are, which causes the filtering approximation for the probabilities
        # to become insufficient.
        print('Starting likelihood: {:.2f}'.format(self.log_likelihood(y)))

        prev_ll = -np.inf
        for i in range(n):
            self.em_once(y, strict=strict)

            ll = self.log_likelihood(y)
            print('\t[{}] ll: {:.2f}'.format(i, ll))
            
            if prev_ll - ll > 0.01:
                print('Error likelihood decreased in EM iteration: {:.4f} -> {:.4f}'.format(prev_ll, ll))
                raise ValueError('Error likelihood decreased')

            prev_ll = ll

    def em_pi(self, em_params):
        st, pxns, p0 = em_params
        return p0

    def em_P(self, em_params):
        st, pxns, p0 = em_params
        return (pxns.sum(0).T / pxns.sum(2).sum(0)).T

    def em_once(self, y, exclude_dists=False, strict=False, em_vars=['pi', 'P']):
        cached = None
        prev_ll = -np.inf

        def smooth(name):
            nonlocal cached, prev_ll
            
            if cached == None or strict == True:
                cached = self.smooth(y)

            if strict == True:
                ll = self.log_likelihood(y)
                
                if prev_ll - ll > 0.01:
                    print('Likelihood decreased before optimizing {}: {:.4f} -> {:.4f} ({:.4f})'.format(name, prev_ll, ll))

                prev_ll = ll

            return cached

        if 'pi' in em_vars:
            self.pi = self.em_pi(smooth('pi'))

        if 'P' in em_vars:
            self.P = self.em_P(smooth('P'))

        if exclude_dists == False:
            st, pxns, p0 = smooth('Dists')
            self.dists = self.em_distributions(st, pxns, y)


class Poisson(sp.stats.rv_discrete):
    def __init__(self, lam):
        self.lam = lam
        self.dist = self

    def params(self):
        return np.array([self.mean()])

    def mean(self):
        return self.lam

    def var(self):
        return self.lam

    def log_pmf(self, k):
        return sp.special.xlogy(k, self.lam) - (self.lam + sp.special.gammaln(k + 1))
    
    def pmf(self, k):
        return np.exp(self.log_pmf(k))


class Normal(sp.stats.rv_continuous):
    def __init__(self, mu, s2):
        self.mu = mu
        self.s2 = s2
        self.dist = self

    def params(self):
        return np.array([self.mean(), self.var()])

    def mean(self):
        return self.mu

    def var(self):
        return self.s2

    def std(self):
        return self.s2 ** 0.5

    def pdf(self, x):
        return util.normal_density(x, self.mu, self.s2)

class PoissonHMM(HMM):
    dist = Poisson

    def __init__(self, P, means, **kwargs):
        HMM.__init__(self, P, self.create_dists(means), **kwargs)
    
    @staticmethod
    def from_flat(P, means, **kwargs):
        return PoissonHMM(P, means, **kwargs)

    def create_dists(self, means):
        return [
            Poisson(m) for m in means
        ]

    def em_distributions(self, st, pxns, y):
        means = (y @ st) / st.sum(0)
        return self.create_dists(means)

class GaussianHMM(HMM):
    dist = Normal

    def __init__(self, P, means, s2, **kwargs):
        HMM.__init__(self, P, self.create_dists(means, s2), **kwargs)

    @staticmethod
    def from_flat(pi, P, params, **kwargs):
        n = params.size // 2
        means = [params[i*2] for i in range(n)]
        s2 = [params[i*2+1] for i in range(n)]
        return GaussianHMM(P, means, s2, **kwargs)

    def create_dists(self, means, s2):
        return [
            Normal(means[i], s2[i]) for i in range(len(means))
        ]

    def em_distributions(self, st, pxns, y):
        sts = st.sum(axis=0)
        means = (y @ st) / sts
        s2 = ((y ** 2 @ st) / sts) - (means ** 2)
        return self.create_dists(means, s2)

class GaussianAR(sp.stats.rv_continuous):
    def __init__(self, alpha, betas, s2):
        self.alpha = alpha
        self.betas = np.array(betas)
        self.s2 = s2
        self.dist = self

    def params(self):
        return np.array([self.alpha, *self.betas, self.s2])

    def mean(self):
        return np.nan

    def var(self):
        return self.s2

    def std(self):
        return self.var() ** 0.5

    def pdf(self, x):
        n = len(self.betas)
        xr = np.lib.stride_tricks.sliding_window_view(x, n, axis=0)[:-1]
        mus = self.alpha + xr @ self.betas
        # The sliding window will chop off the first len(self.betas) - 1 elements, so fill
        # in all these with alpha
        first_mus = np.full(len(self.betas), self.alpha)
        mus = np.concatenate((first_mus, mus))
        e = (x - mus) ** 2 / (2 * self.s2)
        return np.exp(-e) / np.sqrt(2 * np.pi * self.s2)

class GaussianArHMM(HMM):
    dist = GaussianAR

    def __init__(self, P, alphas, betas, s2, **kwargs):
        HMM.__init__(self, P, self.create_dists(alphas, betas, s2), **kwargs)

    def create_dists(self, alphas, betas, s2):
        return [
            GaussianAR(alphas[i], betas[i], s2[i]) for i in range(len(alphas))
        ]

    def em_distributions(self, st, pxns, y):
        alphas = []
        betas = []
        s2s = []

        for i in range(st.shape[1]):
            n = len(self.dists[i].betas)
            w = st[:,i][n:]
            X = np.lib.stride_tricks.sliding_window_view(y, n, axis=0)[:-1]
            X = np.hstack((X, np.expand_dims(np.ones(X.shape[0]), [1])))
            # Compute a weighted least squares regression, using the state probabilities
            # as weights, and adding an intercept
            # p = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y[n:]
            p = np.linalg.pinv(X * w[:, None]) @ (y[n:] * w)
            beta, alpha = p[:-1], p[-1]
            alphas.append(alpha)
            betas.append(beta)

            # Compute the conditional means using our new alphas and betas
            means = X @ p
            resid = y[n:] - means
            s2 = ((resid ** 2) * w).sum() / w.sum()
            s2s.append(s2)

        # alphas = [x.alpha for x in self.dists]
        # s2s = [x.s2 for x in self.dists]
        return self.create_dists(alphas, betas, s2s)

class KalmanMeasurementHMM(KalmanFilter, HMM):
    def __init__(self, y, P, *args, pi=None, **kwargs):
        Z = kwargs['Z']
        kwargs['Z'] = Z[0]
        KalmanFilter.__init__(self, y, **kwargs)
        HMM.__init__(self, P, None, pi=pi)
        self.Z = Z

    def hmm_params(self):
        return self.P, self.pi

    def predict(self):
        xp, ptp, xf, ptf, ks, pts, pys = self.filter()
        yhats = np.array([
            self.Z[i] @ xp for i in range(self.Z.shape[0])
        ])

        pte = np.expand_dims(pts, [2, 3])
        pte = np.swapaxes(pte, 0, 1)
        return (pte * yhats).sum(0)


    def filter(self, pmin=1e-16):
        xp = []
        xf = []
        ptf = []
        ptp = []
        ks = []

        # Pre-compute the state and obs intercepts for the whole series
        xa = self.state_intercept()
        ya = self.obs_intercept()
        xtt = self.x0
        Ptt = self.F @ self.V0 @ self.F.T

        pt = self.pi
        ptt1 = self.P @ pt
        pts = []
        pys = []

        for i,v in enumerate(self.y):
            xtt1, Ptt1 = self.predict_once(i, xtt, Ptt, xa[i], Z=self.Z[0])

            pyv = [self.likelihood_once(i, xtt1, Ptt1, v, ya=ya[j], Z=self.Z[j]) for j in range(self.Z.shape[0])]
            pyv = np.squeeze(np.array(pyv))
            pyv[pyv < pmin] = pmin

            pt, ptt1 = HMM.filter_once(self, ptt1, pyv)

            # Compute the probability weighted average measurement matrix
            pte = np.expand_dims(pt, [1, 2])
            A_avg = (pte * self.Z).sum(0)

            # Filter the next state using the probability weighted average measurement
            # matrix
            xtts = []
            Ptts = []
            Ks = []

            fs = [self.filter_once(i, xtt1, Ptt1, v, ya[i], Z=self.Z[i]) for i in range(self.Z.shape[0])]
            # Invert the list, so that it's aggregated type-wise (i.e. xtts with xtts, Ptts with Ptts, etc)
            fs = zip(*fs)
            xtt, Ptt, K = [(pte * f).sum(0) for f in fs]

            xp.append(xtt1)
            xf.append(xtt)
            ptp.append(Ptt1)
            ptf.append(Ptt)
            ks.append(K)
            pts.append(pt)
            pys.append(pyv)

        return (
            np.array(xp),
            np.array(ptp),
            np.array(xf),
            np.array(ptf),
            np.array(ks),
            np.array(pts),
            np.array(pys)
        )  

    def log_likelihoods(self, **kwargs):
        *filter_params, pt, pys = self.filter()
        lls = [
            self._log_likelihoods(
                filter_params, 
                Z=self.Z[i],
                **kwargs
            ) for i in range(self.Z.shape[0])
        ]

        lls = np.hstack(lls)
        return (pt * lls).sum(1)

    def em_params(self, **kwargs):
        *smooth_params, pts, pys, pxns = self.smooth(**kwargs)

        emps = [
            self._em_params(smooth_params, Z=self.Z[i])
            for i in range(self.Z.shape[0])
        ]

        emps = zip(*emps)
        pte = np.expand_dims(pts, [2])
        pte = np.swapaxes(pte, 0, 1)

        result = []
        
        for v in emps:
            v = np.array(v)
            if v.ndim == 4:
                v = (pte[:,:,None] * v).sum(0)
            elif v.ndim == 3 and v.shape[1] == pte.shape[1]:
                v = (pte * v).sum(0)
            else:
                # These are, e.g. the x0 and V0 parameters
                v = v[0]

            result.append(v)

        result.append(pte)

        return tuple(result)

    def em_Z(self, *args, **kwargs):
        raise ValueError('[KalmanMeasurementHMM] Cannot optimize the measurement matrix (Z) in a measurement HMM')

    def em_once(self, *args, strict=False, starting_likelihood=None, **kwargs):
        cached = None

        def smooth():
            nonlocal cached
            if cached is None or strict == True:
                cached = self.smooth()
            return cached

        smooth_params = None
        if 'em_vars' in kwargs:
            nll = None
            # if 'pi' in em_vars:
            #     *smooth_params, pts, pys = smooth()
            #     p0 = pts[0]
            #     self.pi = self.em_pi((pts, pys, p0))

            if 'P' in kwargs['em_vars']:
                ll = None
                if self.log_level == 'debug':
                    ll = self.log_likelihood()

                *smooth_params, pts, pys, pxns = smooth()
                p0 = pts[0]
                self.P = self.em_P((pts, pxns, p0))

                if self.log_level == 'debug':
                    nll = self.log_likelihood()
                    print('\tP: {:.2f}'.format(nll))
                    if ll > nll:
                        print('[KalmanMeasurementHMM] Likelihood decreased optimizing P: {:.4f} -> {:.4f}'.format(ll, nll))

                    starting_likelihood = nll

        # Remove the default inclusion of the measurement matrix as a parameter to be optimized
        return KalmanFilter.em_once(self, *args, **kwargs, strict=strict, starting_likelihood=starting_likelihood)

    def em_x0(self, em_params, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params(**kwargs)        
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, pts = em_params

        if 'x0' in self.constraints:
            f, D = self.constraints['x0']
        else:
            f = np.zeros((1, x0.shape[0], 1))
            D = np.eye(x0.shape[0])[None]

        nums, denoms = zip(*[
            self._em_x0_degenerate(em_params, f, D, **kwargs, Z=self.Z[i])
            for i in range(self.Z.shape[0])
        ])

        pts = pts[:,:,:,None]
        num = np.array(nums)
        denom = np.array(denoms)

        num = (pts * num).sum(0).sum(0)
        denom = (pts * denom).sum(0).sum(0)

        v = util.safe_inverse(denom) @ num
        v = f + D @ v
        x0_ = v.reshape(x0.shape)

        return x0_

    def em_U(self, em_params, **kwargs):
        B, Z, U, Q, R, A, x0, V0, G, H, F, state_exog, obs_exog = self.params(**kwargs)        
        x, xx, pt, s11, s10, s00, e, xa, ya, x0_, V0_, resid_cov, pts = em_params

        if 'U' in self.constraints:
            f, D = self.constraints['U']
        else:
            Usz = x0.shape[0] * state_exog.shape[1]
            f = np.zeros((1, Usz, 1))
            D = np.eye(Usz)[None]

        nums, denoms = zip(*[
            self._em_U_degenerate(em_params, f, D, **kwargs, Z=self.Z[i])
            for i in range(self.Z.shape[0])
        ])

        pts = pts[:,:,:,None]
        num = np.array(nums)
        denom = np.array(denoms)
        num = (pts * nums).sum(0).sum(0)
        denom = (pts * denoms).sum(0).sum(0)

        u = util.safe_inverse(denom) @ num
        u = f + D @ u
        U = u.reshape(U.shape)

        return U

    def smooth_cov_init(self, pt, K, Pn1):
        covs = [KalmanFilter.smooth_cov_init(self, K, Pn1, Z=self.Z[i]) for i in range(self.Z.shape[0])]
        covs = np.array(covs)
        pt = np.expand_dims(pt, [1, 2])
        return (pt * covs).sum(0)

    def smooth(self, filter_params=None):
        # The smoother is too computationally intensive, so just use the filter as an approximation
        *filter_params, pts, pys = self.filter() if filter_params is None else filter_params
        pxns = [np.outer(pts[i], pts[i+1]) for i in range(pts.shape[0] - 1)]
        pxns.append(self.smooth_init_pxn(pts[-1]))
        pxns = np.array(pxns)

        cov_init = self.smooth_cov_init(pts[-1], filter_params[-1][-1], filter_params[-2][-2])
        return *self._smooth(filter_params, Z=self.Z[0], cov_init=cov_init), pts, pys, pxns

    def named_params(self):
        p = KalmanFilter.named_params(self)
        p['P'] = self.P
        p['pi'] = self.pi
        return p

    # @classmethod
    # def unflatten_params(cls, params, state_dim, state_exog_dim, obs_dim, obs_exog_dim, args, constraints, i=0):
    #     hmm_states = len(args['A'])

    #     if 'pi' in args:
    #         pi = args['pi']
    #     else:
    #         pi, i = cls.extract_parameter(i, params, (hmm_states, 1), None)

    #     if 'P' in args:
    #         P = args['P']
    #         P, i = cls.extract_parameter(i, params, (hmm_states, hmm_states), None)

    #     if 'A' not in args:
    #         args['A'], i = cls.extract_parameter(i, params, (hmm_states, obs_dim, state_dim), constraints['A'] if 'A' in constraints else None)

    #     return KalmanFilter.unflatten_params(params, state_dim, state_exog_dim, obs_dim, obs_exog_dim, args, constraints, i=i)
        

def kf_hmm_expected(params, pt, fn):
    vals = kf_hmm_eval(params, fn)

    # If pt's ndim is 1, then we're dealing with a single state tuple and we want to average it. If pt's ndim is 2
    # then we're dealing with a vector of state tuples, and we want to average over each
    # of the state tuples.
    sum_axis = 0 if pt.ndim == 1 else 1
    pt = np.expand_dims(pt, [1, 2]) if pt.ndim == 1 else np.expand_dims(pt, [2])

    if type(vals[0]) == tuple:
        # Invert the pairings, so we're now grouping by type
        # rather than state
        vals = list(zip(*vals))
        res = []
        
        for v in vals:
            v = np.hstack(v)
            v = (pt * v).sum(sum_axis)
            v = np.expand_dims(v, [2])
            res.append(v)

        return tuple(res)
    else:
        vals = np.array(vals)
        return (pt * vals).sum(sum_axis)


def kf_hmm_em_params(pt, params, y):
    filter_params = kf_filter(params, y)
    smooth_params = kf_smooth(params, filter_params)
    return kf_hmm_expected(params, lambda st_params: kf_em_params_(st_params, yz))

def kf_hmm_em_R(pt, params, y, em_params, constraints=()):
    Rs = kf_hmm_eval(params, lambda st_params: kf_em_R(st_params, y, em_params, constraints=constraints))
    Rs = np.array(Rs)
    return (pt * covs).sum()

def create_hmm(params, state_dim, cls):
    i = 0
    
    pi = params[i:i+state_dim]
    i += state_dim

    ssdim = state_dim ** 2
    Phi = params[i:i+ssdim].reshape((state_dim, state_dim))
    i += ssdim

    pi = np.abs(pi)
    pi /= pi.sum()

    Phi = np.abs(Phi)
    Phi = (Phi.T / Phi.sum(1)).T

    return cls.from_flat(Phi, params[i:], pi=pi)