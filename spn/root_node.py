import copy

from .node import Node

class RootNode(Node):
	def __init__(self, node):
		super(RootNode, self).__init__(node.n, node.scope)
		self.children.append(node)
		node.parent = self

	def evaluate(self, obs):
		return self.children[0].evaluate(obs)

	def update(self, obs, params):
		self.children[0].update(obs, params)
		self.n += 1
		if params.prunebatch > 0 and self.n % params.prunebatch == 0:
			params = copy.copy(params)
			params.updatestruct = False
			self.children[0].prune(params.maxdepth, params)

	def display(self, depth=0):
		self.children[0].display(depth)

	def add_child(self, child):
		assert len(self.children) == 0
		self.children.append(child)
		child.parent = self

	def remove_child(self, child):
		assert len(self.children) == 1
		self.children.remove(child)
		child.parent = None

