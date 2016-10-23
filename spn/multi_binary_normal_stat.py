import math

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import mvn

from .multi_normal_stat import MultiNormalStat

def make_bounds(x):
	zind = x==0
	lo = np.where(zind, -3.5, 0.5)
	hi = np.where(zind, 0.5, 4.5)
	return lo, hi

class MultiBinaryNormalStat:

	@staticmethod
	def create(nvar):
		stat = MultiBinaryNormalStat()
		stat.normal = MultiNormalStat.create(nvar)
		return stat

	@staticmethod
	def create_copy(mean, cov):
		stat = MultiBinaryNormalStat()
		stat.normal = MultiNormalStat.create_copy(mean, cov)
		return stat

	def __repr__(self):
		return "BinaryNormal({0}, {1})".format(self.normal.mean, self.normal.cov.flatten())

	def evaluate(self, x):
		lo, hi = make_bounds(x)
		p, i = mvn.mvnun(lo, hi, self.normal.mean, self.normal.cov+0.0001*np.identity(len(x)))
		return math.log(np.clip(p,0.0001,0.9999))

	def update(self, x, n):
		self.normal.update(x, n)

	def iterate_corrs(self, corrthresh):
		for i, j in self.normal.iterate_corrs(corrthresh):
			yield i, j

	def distill(self):
		return self.normal.distill()

	def extract(self, ind):
		return MultiBinaryNormalStat.create_copy(self.normal.mean[ind], self.normal.cov[np.ix_(ind,ind)])

