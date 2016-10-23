import numpy as np
from scipy.misc import logsumexp

from .node import Node

class SumNode(Node):
	def __init__(self, n, scope):
		super(SumNode, self).__init__(n, scope)

	def display(self, depth=0):
		print("{0}<+ {1} {2}>".format('-'*depth, self.n, self.scope))
		for child in self.children:
			child.display(depth+1)

	def evaluate(self, obs):
		logprobs = self.evaluate_children(obs, False)
		return logsumexp(logprobs)

	def evaluate_children(self, obs, equal_weight):
		logprobs = np.array([child.evaluate(obs) for child in self.children])
		if equal_weight:
			return logprobs
		else:
			logweights = np.array([self.get_log_weight(child) for child in self.children])
			return logprobs + logweights			

	def get_log_weight(self, child):
		return np.log(child.n) - np.log(self.n)

	def update(self, obs, params):
		logprobs = self.evaluate_children(obs, params.equalweight)
		topchild = self.children[np.argmax(logprobs)]
		self.n += 1
		topchild.update(obs, params)

	def add_child(self, child):
		assert np.array_equal(child.scope, self.scope)
		child.parent = self
		self.children.append(child)

	def remove_child(self, child):
		self.children.remove(child)
		child.parent = None

	def prune(self, depth, params):
		nc = len(self.children)
		for i in range(nc-1, -1, -1):
			child = self.children[i]
			if child.n == 1:
				obs = child.rep()
				self.children.pop(i)
				self.n -= 1
				self.update(obs, params)
		for child in self.children:
			child.prune(depth-1, params)

