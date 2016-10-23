import copy

from .node import Node
from .multi_normal_stat import MultiNormalStat

class MultiNormalLeafNode(Node):

	@staticmethod
	def create(n, scope):
		node = MultiNormalLeafNode(n, scope)
		node.stat = MultiNormalStat.create(len(scope))
		return node

	@staticmethod
	def create_from_stat(n, scope, stat):
		node = MultiNormalLeafNode(n, scope)
		node.stat = copy.deepcopy(stat)
		return node

	def __repr__(self):
		return '<" {0} {1} {2}>'.format(self.n, self.scope, self.stat)

	def display(self, depth=0):
		print('{0}{1}'.format('-'*depth, self))

	def rep(self):
		obs = np.empty(np.max(self.scope)+1)
		obs[self.scope] = self.stat.rep()
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
	mean = np.array([1., 2., 3.])
	cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
	obs = np.random.multivariate_normal(mean, cov, 100000)
	node = MultiNormalLeafNode.create(0, np.array([0,1]))
	for x in obs:
		node.update(x, None)
	print(node.evaluate(obs[0]))
#	print(node.n)
#	print(node.scope)
#	print(node.mvn.mean)
#	print(node.mvn.cov)
#	node.display()
