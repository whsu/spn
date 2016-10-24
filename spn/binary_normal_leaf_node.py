import numpy as np
import math

from .node import Node

class BinaryNormalLeafNode(Node):
	def __init__(self, n, index, p=0.0):
		super(BinaryNormalLeafNode, self).__init__(n, np.array([index]))
		self.index = index
		self.p = p

	def display(self, depth=0):
		print("{0}<' {1} {2} {3:.3f}>".format(
		        '-'*depth, self.n, self.scope, self.p))

	def rep(self):
		obs = np.empty(self.index+1)
		obs[self.index] = 1 if self.p > 0.5 else 0
		return obs

	def evaluate(self, obs):
		p = np.clip(self.p, 0.0001, 0.9999)
		if obs[self.index] == 1:
			return math.log(p)
		else:
			return math.log(1-p)

	def update(self, obs, params):
		self.p = (self.n*self.p+obs[self.index])/(self.n+1)
		self.n += 1

	def prune(self, depth, params):
		pass

	@staticmethod
	def create_from_stat(n, scope, stat):
		nodes = [None] * len(scope)
		for i in range(len(scope)):
			v = scope[i]
			a = tuple(k for k in range(len(scope)) if k != i)
			nodes[i] = BinaryNormalLeafNode(n, v, stat.normal.mean[i])
		return nodes

