import copy

from .node import Node
from .multi_binary_normal_stat import MultiBinaryNormalStat

class MultiBinaryNormalLeafNode(Node):

	@staticmethod
	def create(n, scope):
		node = MultiBinaryNormalLeafNode(n, scope)
		node.stat = MultiBinaryNormalStat.create(len(scope))
		return node

	@staticmethod
	def create_from_stat(n, scope, stat):
		node = MultiBinaryNormalLeafNode(n, scope)
		node.stat = copy.deepcopy(stat)
		return node

	def __repr__(self):
		return '<" {0} {1} {2}>'.format(self.n, self.scope, self.stat)

	def display(self, depth=0):
		print('{0}{1}'.format('-'*depth, self))

	def rep(self):
		obs = np.empty(np.max(self.scope)+1, dtype=int)
		obs[self.scope] = self.stat.distill()
		return obs

	def evaluate(self, obs):
		x = obs[self.scope]
		return self.stat.evaluate(x)

	def update(self, obs, params):
		self.stat.update(obs[self.scope], self.n)
		self.n += 1

	def prune(self, depth, params):
		pass

if __name__ == '__main__':
	import numpy as np
	from scipy.stats import multivariate_normal

	np.set_printoptions(precision=3)
	np.random.seed(0)

	n = 10000
	x = np.random.binomial(1, 0.1, (n,1))
	y = np.empty((n,1))
	i = np.where(x==0)[0]
	j = np.where(x==1)[0]
	y[i] = np.random.binomial(1, 0.2, (len(i),1))
	y[j] = np.random.binomial(1, 0.9, (len(j),1))
	z = np.random.binomial(1, 0.8, (n,1))
	obs = np.hstack((x,y,z)).astype(int)

	node = MultiBinaryNormalLeafNode.create(0, np.array([0,1,2]))
	for x in obs:
		node.update(x, None)
	node.display()
