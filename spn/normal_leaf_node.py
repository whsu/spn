import numpy as np
from scipy.stats import norm
import math

from .node import Node

class NormalLeafNode(Node):
	def __init__(self, n, index, mean=0.0, var=1.0):
		super(NormalLeafNode, self).__init__(n, np.array([index]))
		self.index = index
		self.mean = mean
		self.var = var

	def display(self, depth=0):
		print("{0}<' {1} {2} {3} {4}>".format(
		        '-'*depth, self.n, self.scope, self.mean, self.var))

	def logpdf(self, x):
		d = x - self.mean
		v2 = 2 * self.var
		return -d*d/v2 - 0.5*math.log(v2*math.pi)

	def rep(self):
		obs = np.empty(self.index+1)
		obs[self.index] = self.mean
		return obs

	def evaluate(self, obs):
		return self.logpdf(obs[self.index])

	def update(self, obs, params):
		x = obs[self.index]
		mean = (self.n*self.mean + x) / (self.n + 1)
		var = (self.n*self.var + (x-self.mean)*(x-mean)) / (self.n + 1)
		self.n += 1
		self.mean = mean
		self.var = var

	def prune(self, depth, params):
		pass

	@staticmethod
	def create_from_stat(n, scope, stat):
		nodes = [None] * len(scope)
		for i in range(len(scope)):
			v = scope[i]
			nodes[i] = NormalLeafNode(n, v, stat.mean[i], stat.cov[i,i])
		return nodes

if __name__ == '__main__':
	from .spn import *
	obs = np.random.normal(1.0, 2.0, (10000,3))
	node = NormalLeafNode(0, 1, 0.0, 0.0)
	params = SPNParams()
	for x in obs:
		node.update(x, params)
	print(node.n)
	print(node.scope)
	print(node.mean)
	print(node.var)	
