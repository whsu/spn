import numpy as np
from scipy.stats import multivariate_normal

class MultiNormalStat:

	@staticmethod
	def create(nvar):
		stat = MultiNormalStat()
		stat.mean = np.zeros(nvar)
		stat.cov = np.identity(nvar)
		return stat

	@staticmethod
	def create_copy(mean, cov):
		stat = MultiNormalStat()
		stat.mean = mean.copy()
		stat.cov = cov.copy()
		return stat

	def __repr__(self):
		return "Normal({0}, {1})".format(self.mean, self.cov.flatten())

	def evaluate(self, x):
		return multivariate_normal.logpdf(x, self.mean, self.cov)

	def update(self, x, n):
		mean = self.mean
		cov = self.cov

		mean_new = (n * mean + x) / (n + 1)

		dx = x - mean
		dm = mean_new - mean

		cov_new = np.empty_like(cov)
		for i in range(len(x)):
			for j in range(len(x)):
				cov_new[i,j] = (n*cov[i,j]+dx[i]*dx[j]-(n+1)*dm[i]*dm[j])/(n+1)

		self.mean = mean_new
		self.cov = cov_new

	def iterate_corrs(self, corrthresh):
		v = np.diag(self.cov)
		corrs = np.abs(self.cov) / np.sqrt(np.outer(v, v))
		rows, cols = np.unravel_index(np.argsort(corrs.flatten()), corrs.shape)

		for i, j in zip(reversed(rows), reversed(cols)):
			if corrs[i, j] < corrthresh:
				break
			yield i, j

	def distill(self):
		return self.mean

	def extract(self, ind):
		return MultiNormalStat.create_copy(self.mean[ind], self.cov[np.ix_(ind,ind)])

	def extract_from_obs(self, ind, x):
		cov = self.cov[np.ix_(ind,ind)]
		stat = MultiNormalStat.create_copy(x, np.diag(np.diag(cov)))
		return stat

