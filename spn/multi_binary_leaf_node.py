import copy

from .node import Node
from .multi_binary_stat import MultiBinaryStat

class MultiBinaryLeafNode(Node):

	@staticmethod
	def create(n, scope):
		node = MultiBinaryLeafNode(n, scope)
		node.stat = MultiBinaryStat.create(len(scope))
		return node

	@staticmethod
	def create_from_stat(n, scope, stat):
		node = MultiBinaryLeafNode(n, scope)
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

