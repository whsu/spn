import math
import numpy as np

def phi_correlation(m):
	si = np.sum(m, axis=1)
	sj = np.sum(m, axis=0)
	return abs(m[1,1]*m[0,0]-m[1,0]*m[0,1])/math.sqrt(si[0]*si[1]*sj[0]*sj[1])

class MultiBinaryStat:

	@staticmethod
	def create(nvar):
		stat = MultiBinaryStat()
		stat.probs = np.zeros(nvar)
		stat.matrices = {}
		for i in range(nvar):
			for j in range(i+1, nvar):
				stat.matrices[(i,j)] = 0.25*np.ones((2,2))
		return stat

	@staticmethod
	def create_copy(probs, matrices):
		stat = MultiBinaryStat()
		stat.probs = probs.copy()
		stat.matrices = matrices
		return stat

	def __repr__(self):
		return "Binary({0}, {1})".format(self.probs, self.matrices)

	def evaluate(self, x):
		assert issubclass(x.dtype.type, np.integer)
		assert len(self.matrices) == 1
		return math.log(np.clip(self.matrices[(0,1)][x[0]][x[1]], 0.0001, 0.9999))

	def update(self, x, n):
		nvar = len(x)
		self.probs = (n*self.probs+x) / (n+1)
		for i in range(nvar):
			for j in range(i+1, nvar):
				matrix = np.zeros((2,2))
				matrix[x[i],x[j]] = 1
				self.matrices[(i,j)] = ((n+1)*self.matrices[(i,j)] + matrix) / (n+2)

	def iterate_corrs(self, corrthresh):
		corrs = {pair:phi_correlation(self.matrices[pair]) for pair in self.matrices}
		indices = sorted(corrs, key=lambda p:-corrs[p])
		for ind in indices:
			if corrs[ind] < corrthresh:
				break
			yield ind

	def distill(self):
		return (self.probs>0.5).astype(int)

	def extract(self, ind):
		nvar = len(ind)
		matrices = {}
		for i in range(nvar):
			for j in range(i+1, nvar):
				matrices[(i,j)] = self.matrices[(ind[i],ind[j])]
		return MultiBinaryStat.create_copy(self.probs[ind], matrices)

	def extract_from_obs(self, ind, x):
		stat = MultiBinaryStat.create(len(ind))
		stat.update(x, 1)
		return stat

