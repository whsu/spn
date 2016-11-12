import numpy as np

from .root_node import RootNode
from .product_node import ProductNode
from .normal_leaf_node import NormalLeafNode
from .multi_normal_leaf_node import MultiNormalLeafNode

class SPNParams:
	"""
	Parameters
	----------
	batchsize : number of samples in a mini-batch.
	mergebatch : number of samples a product node needs to see before updating
	             its structure.
	corrthresh : correlation coefficient threshold above which two variables
	             are considered correlated.
	equalweight : whether sum nodes should consider children as having equal
	              weights when deciding which children to pass data to.
	updatestruct : whether to update the network structure.
	maxdepth : depth at which to prune the tree.
	mvleaf : whether to use multivariate leaves.
	mvmaxscope : number of variables that can be combined into a multivariate
	             leaf node.
	leaftype : type of leaf nodes, one of "normal", "binary", "binarynormal".
	"""
	def __init__(self, batchsize=128, mergebatch=128, corrthresh=0.1,
	             equalweight=True, updatestruct=True, maxdepth=20,
	             mvleaf=True, mvmaxscope=2, leaftype="normal"):
		if leaftype != "normal":
			raise ValueError("Leaf type {0} not supported.".format(leaftype))
		self.batchsize = batchsize
		self.mergebatch = mergebatch
		self.corrthresh = corrthresh
		self.equalweight = equalweight
		self.updatestruct = updatestruct
		self.maxdepth = maxdepth
		self.mvleaf = mvleaf
		self.mvmaxscope = mvmaxscope
		self.leaftype = leaftype
		self.binary = False if leaftype=="normal" else True

class SPN:
	"""
	Parameters
	----------
	node : int or Node
		if int, number of variables
		if Node, root of network
	params : SPNParams
		parameters of the network
	"""
	def __init__(self, node, params):
		if type(node) == int:
			numvar = node
			scope = np.arange(numvar)
			node = make_product_net(scope, params.leaftype)
		self.root = RootNode(node)
		self.params = params

	def evaluate(self, obs):
		if obs.ndim == 1:
			obs = obs.reshape(1, len(obs))
		return self.root.evaluate(obs)

	def update(self, obs):
		if obs.ndim == 1:
			obs = obs.reshape(1, len(obs))
		for i in range(0, len(obs), self.params.batchsize):
			self.root.update(obs[i:i+self.params.batchsize], self.params)

	def display(self):
		self.root.display()

def make_product_net(scope, leaftype):
	node = ProductNode(0, scope, leaftype)
	for v in scope:
		node.add_child(node.Leaf(0, v))
	return node

