import numpy as np
import scipy as sp
from . import util
from hmmlearn.hmm import GaussianHMM as hl_GaussianHMM

def cloneHmmLearn(gh):
    model = hl_GaussianHMM(n_components=gh.Phi.shape[0], covariance_type='diag')
    model.startprob_ = gh.pi
    model.transmat_ = gh.Phi
    model.means_ = np.expand_dims(np.array([x.mean() for x in gh.dists]), [1])
    model.covars_ = np.expand_dims(np.array([x.var() for x in gh.dists]), [1])
    return model

class HMM:
	def __init__(self, Phi, dists, pi=None):
		self.dists = dists
		# self.dists = [sp.stats.poisson(mu) for mu in state_params]
		# self.state_params = state_params
		self.Phi = Phi
		self.pi = util.solve_stationary(Phi.T) if type(pi) == type(None) else pi
		self.state_dim = self.Phi.shape[0]

	def filter(self, y):
		# My textbook and other implementations seem to set ptt1 = self.pi rather than pt,
		# but the EM algorithm decreases in likelihood occasionally when I do it that way, and
		# this way it does not. It also seems more logical this way to me.
		pt = self.pi
		pred = []
		filtered = []

		py = self.pdf(y).T

		for i,v in enumerate(y):
			ptt1 = pt @ self.Phi
			state = np.zeros(self.state_dim)
			pt = (py[i] * ptt1) / (py[i] @ ptt1)
			
			pred.append(ptt1)
			filtered.append(pt)

		return np.array(pred), np.array(filtered)

	def pdf(self, val):
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

		return np.array(result)

	def smooth(self, y, eps=1e-12):
		ptp, ptf = self.filter(y)
		pnt = ptf[-1]
		states = [pnt]
		py = self.pdf(y).T
		p = np.ones(self.pi.shape)
		pxns = []
		# pxns = [((1 / p) * pnt * (self.Phi * py).T).T]

		for i,v in enumerate(reversed(y[:-1])):
			# We only care about using the relative weights of p, and if we alllow it to recursively
			# multiply backwards, it just keeps shrinking in scale. So just periodically rescale it.
			if p.min() < 1e-15 or p.max() > 1e15:
				prevp = prevp / prevp.sum()
				p = p / p.sum()

			prevp = p
			pyv = py[-(i + 1)]
			p = self.Phi @ (pyv * p)

			ptt = ptf[-(i + 2)]
			pnt = (ptt * p) / (ptt @ p)
			pxn = (pnt * (self.Phi * pyv * prevp).T / p).T
			
			states.append(pnt)
			pxns.append(pxn)

		prevp = p
		p = self.Phi @ (py[0] * p)
		pn0 = (self.pi * p) / (self.pi @ p)

		states = list(reversed(states))
		pxns = list(reversed(pxns))

		return np.array(states), np.array(pxns), pn0

	def sample(self, n):
		svals = np.random.uniform(0, 1, size=n)
		st = np.where(svals[0] <= self.pi.cumsum())[0][0]
		states = [st]
		vals = [self.dists[st].rvs(1)[0]]

		for sv in svals[1:]:
			st = np.where(sv <= self.Phi[st].cumsum())[0][0]
			states.append(st)
			vals.append(self.dists[st].rvs(1)[0])

		return np.array(states), np.array(vals)

	def log_likelihood(self, y):
		ptp, ptf = self.filter(y)
		py = np.squeeze(self.pdf(y))
		dv = (ptp * py.T).sum(axis=1) # Probability weighted sum of the masses
		return np.log(dv).sum()

	def minimize(self, y, method='BFGS'):
		params = np.concatenate((
			self.pi,
			self.Phi.flatten(),
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
		self.Phi = hmm.Phi
		self.dists = hmm.dists
		print('Minimized: {:.2f}'.format(self.log_likelihood(y)))


	def em_distributions(self, st, pxns, y):
		# This is to be overriden by distributional specific child classes
		pass

	def em(self, y, n=10, strict=False):
		print('Starting likelihood: {:.2f}'.format(self.log_likelihood(y)))

		prev_ll = -np.inf
		for i in range(n):
			self.em_iter(y, strict=strict)

			ll = self.log_likelihood(y)
			print('\t[{}] ll: {:.2f}'.format(i, ll))
			
			if prev_ll - ll > 0.01:
				print('Error likelihood decreased in EM iteration: {:.4f} -> {:.4f}'.format(prev_ll, ll))
				raise ValueError('Error likelihood decreased')

			prev_ll = ll

	def em_iter(self, y, exclude_dists=False, strict=False):
		cached = None
		prev_ll = -np.inf

		def smooth(name):
			nonlocal cached, prev_ll
			
			if cached == None or strict == True:
				cached = self.smooth(y)

			if strict == True:
				ll = self.log_likelihood(y)
				
				if prev_ll - ll > 0.01:
					old = self.Phi
					self.Phi = (self.Phi.T / self.Phi.sum(1)).T
					pll = self.log_likelihood(y)
					self.Phi = old
					print('Likelihood decreased before optimizing {}: {:.4f} -> {:.4f} ({:.4f})'.format(name, prev_ll, ll, pll))
					# raise ValueError('test')

				prev_ll = ll

			return cached

		st, pxns, p0 = smooth('pi')
		self.pi = p0

		st, pxns, p0 = smooth('Phi')
		self.Phi = (pxns.sum(0).T / pxns.sum(2).sum(0)).T

		if exclude_dists == False:
			st, pxns, p0 = smooth('Dists')
			self.dists = self.em_distributions(st, pxns, y)
			smooth('End')


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
		e = (x - self.mu) ** 2 / (2 * self.s2)
		return np.exp(-e) / np.sqrt(2 * np.pi * self.s2)

class PoissonHMM(HMM):
	dist = Poisson

	def __init__(self, Phi, means, **kwargs):
		HMM.__init__(self, Phi, self.create_dists(means), **kwargs)
	
	@staticmethod
	def from_flat(Phi, means, **kwargs):
		return PoissonHMM(Phi, means, **kwargs)

	def create_dists(self, means):
		return [
			Poisson(m) for m in means
		]

	def em_distributions(self, st, pxns, y):
		means = (y @ st) / st.sum(0)
		return self.create_dists(means)

class GaussianHMM(HMM):
	dist = Normal

	def __init__(self, Phi, means, s2, **kwargs):
		HMM.__init__(self, Phi, self.create_dists(means, s2), **kwargs)

	@staticmethod
	def from_flat(pi, Phi, params, **kwargs):
		n = params.size // 2
		means = [params[i*2] for i in range(n)]
		s2 = [params[i*2+1] for i in range(n)]
		return GaussianHMM(Phi, means, s2, **kwargs)

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
		return np.nanfs

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

	def __init__(self, Phi, alphas, betas, s2, **kwargs):
		HMM.__init__(self, Phi, self.create_dists(alphas, betas, s2), **kwargs)

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

		return self.create_dists(alphas, betas, s2s)

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
