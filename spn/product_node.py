import numpy as np

from .node import Node
from .sum_node import SumNode
from .normal_leaf_node import NormalLeafNode
from .multi_normal_leaf_node import MultiNormalLeafNode
from .multi_normal_stat import MultiNormalStat
from .binary_normal_leaf_node import BinaryNormalLeafNode
from .multi_binary_normal_leaf_node import MultiBinaryNormalLeafNode
from .multi_binary_normal_stat import MultiBinaryNormalStat
from .binary_leaf_node import BinaryLeafNode
from .multi_binary_leaf_node import MultiBinaryLeafNode
from .multi_binary_stat import MultiBinaryStat

class ProductNode(Node):
	def __init__(self, n, scope, leaftype, src=None):
		super(ProductNode, self).__init__(n, scope)
		if leaftype == "binary":
			self.Stat = MultiBinaryStat
			self.Leaf = BinaryLeafNode
			self.MVLeaf = MultiBinaryLeafNode
			self.dtype = int
		elif leaftype == "binarynormal":
			self.Stat = MultiBinaryNormalStat
			self.Leaf = BinaryNormalLeafNode
			self.MVLeaf = MultiBinaryNormalLeafNode
			self.dtype = int
		else:
			self.Stat = MultiNormalStat
			self.Leaf = NormalLeafNode
			self.MVLeaf = MultiNormalLeafNode
			self.dtype = float
		m = len(scope)
		if src is None:
			self.stat = self.Stat.create(m)
			self.merges = 0
		else:
			ind = [src.v2i[k] for k in scope]
			self.stat = src.stat.extract(ind)
			self.merges = src.merges
		self.i2c = [None] * len(scope)  # variable index to child mapping
		self.v2i = {k:i for i, k in enumerate(scope)} # variable to index mapping

	def display(self, depth=0):
		print("{0}<* {1} {2}>".format('-'*depth, self.n, self.scope))
		for child in self.children:
			child.display(depth+1)

	def rep(self):
		obs = np.empty(np.max(self.scope)+1, dtype=self.dtype)
		obs[self.scope] = self.stat.distill()
		return obs

	def evaluate(self, obs):
		value = 0.0
		for child in self.children:
			value += child.evaluate(obs)
		return value

	def update(self, obs, params):
		self.stat.update(obs[self.scope], self.n)

		merged = False
		if params.updatestruct and self.n >= params.mergebatch*(self.merges+1):
			for i, j in self.stat.iterate_corrs(params.corrthresh):
				if self.i2c[i] == self.i2c[j]:
					continue
				self.merge_children(i, j, obs, params)
				merged = True
				break

		if not merged:
			self.update_children(obs, params)

		self.n += 1

	def update_children(self, obs, params):
		for child in self.children:
			child.update(obs, params)

	def add_child(self, child):
		for v in child.scope:
			i = self.v2i[v]
			assert self.i2c[i] is None
			self.i2c[i] = child
		child.parent = self
		self.children.append(child)

	def remove_child(self, child):
		for v in child.scope:
			i = self.v2i[v]
			assert self.i2c[i] is not None
			self.i2c[i] = None
		self.children.remove(child)
		child.parent = None

	def merge_into_mvleaf(self, ci, cj, scope, obs, params):
		ind = [self.v2i[k] for k in scope]
		m = self.MVLeaf.create_from_stat(self.n+1, scope, self.stat.extract(ind))
		self.remove_children(ci, cj)
		if len(self.children) > 0:
			self.update_children(obs, params)
			self.add_child(m)
		else:
			parent = self.parent
			parent.remove_child(self)
			parent.add_child(m)

	def map_scope(self, scope):
		return np.array([self.v2i[v] for v in scope])

	def merge_into_sumnode(self, ci, cj, scope, obs, params):
		p1 = ProductNode(self.n, scope, params.leaftype, self)
		p1.add_children(ci, cj)

		p2 = ProductNode(1, scope, params.leaftype)
		p2.stat = self.stat.extract_from_obs(self.map_scope(scope), obs[scope])
		children = self.Leaf.create_from_stat(p2.n, p2.scope, p2.stat)
		p2.add_children(*children)

		s = SumNode(p1.n+p2.n, scope)
		s.add_children(p1, p2)

		self.remove_children(ci, cj)
		if len(self.children) > 0:
			self.update_children(obs, params)
			self.add_child(s)
		else:
			parent = self.parent
			parent.remove_child(self)
			parent.add_child(s)
			if type(parent) == SumNode:
				s.remove_children(p1, p2)
				parent.remove_child(s)
				parent.add_children(p1, p2)

	def merge_children(self, i, j, obs, params):
		self.merges += 1
		ci = self.i2c[i]
		cj = self.i2c[j]
		scope = np.concatenate((ci.scope, cj.scope))
		scope.sort()

		if len(scope) <= params.mvmaxscope:
			self.merge_into_mvleaf(ci, cj, scope, obs, params)
		else:
			self.merge_into_sumnode(ci, cj, scope, obs, params)

	def prune(self, depth, params):
		if depth > 0:
			for child in self.children:
				child.prune(depth-1, params)
		elif params.mvleaf:
			node = self.MVLeaf.create_from_stat(self.n, self.scope, self.stat)
			parent = self.parent
			parent.remove_child(self)
			parent.add_child(node)
		else:
			self.children = []
			self.i2c = [None] * len(self.scope)
			children = self.Leaf.create_from_stat(self.n, self.scope, self.stat)
			self.add_children(*children)

if __name__ == '__main__':
	from .normal_leaf_node import *
	from .spn import *
	np.set_printoptions(precision=3, suppress=True)
	np.random.seed(0)
	n = 10000

#	mean = np.array([1., 2., 3.])
#	cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
#	obs = np.random.multivariate_normal(mean, cov, n)
#	child1 = NormalLeafNode(0, 0)
#	child2 = NormalLeafNode(0, 1)
#	child3 = NormalLeafNode(0, 2)
#	node = ProductNode(0, np.array([0,1,2]), binary=False)

	n = 10000
	x = np.random.binomial(1, 0.1, (n,1))
	y = np.empty((n,1))
	i = np.where(x==0)[0]
	j = np.where(x==1)[0]
	y[i] = np.random.binomial(1, 0.2, (len(i),1))
	y[j] = np.random.binomial(1, 0.9, (len(j),1))
	z = np.random.binomial(1, 0.8, (n,1))
	obs = np.hstack((x,y,z)).astype(int)

	child1 = BinaryLeafNode(0, 0)
	child2 = BinaryLeafNode(0, 1)
	child3 = BinaryLeafNode(0, 2)
	node = ProductNode(0, np.array([0,1,2]), binary=True)

	node.add_children(child1, child2, child3)
	s = SPN(node, SPNParams(mergebatch=1000, maxdepth=5))
	i = 0
	for x in obs:
		s.update(x)
		i += 1
		if i % 100 == 0:
			print(i)
	s.display()

