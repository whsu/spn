import numpy as np

from .root_node import RootNode
from .product_node import ProductNode
from .normal_leaf_node import NormalLeafNode
from .multi_normal_leaf_node import MultiNormalLeafNode

class SPNParams:
	def __init__(self, batchsize=128, mergebatch=128, corrthresh=0.1, equalweight=True,
	             updatestruct=True, maxdepth=20, prunebatch=1000, mvleaf=True,
	             mvmaxscope=2, leaftype="normal"):
		assert not (mvmaxscope > 2 and leaftype=="binary")
		self.batchsize = batchsize
		self.mergebatch = mergebatch
		self.corrthresh = corrthresh
		self.equalweight = equalweight
		self.updatestruct = updatestruct
		self.maxdepth = maxdepth
		self.prunebatch = prunebatch
		self.mvleaf = mvleaf
		self.mvmaxscope = mvmaxscope
		self.leaftype = leaftype
		self.binary = False if leaftype=="normal" else True

class SPN:
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

if __name__ == '__main__':
	from .product_node import ProductNode
	from .multi_normal_leaf_node import MultiNormalLeafNode
	from .normal_leaf_node import NormalLeafNode
	import numpy as np

	np.set_printoptions(precision=3)
	np.random.seed(0)
	n = 10000

	mean = np.array([1., 2., 3.])
	cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
	obs = np.random.multivariate_normal(mean, cov, n)
	s = SPN(3, SPNParams(mvmaxscope=0))

#	x = np.random.binomial(1, 0.1, (n,1))
#	y = np.empty((n,1))
#	i = np.where(x==0)[0]
#	j = np.where(x==1)[0]
#	y[i] = np.random.binomial(1, 0.2, (len(i),1))
#	y[j] = np.random.binomial(1, 0.9, (len(j),1))
#	z = np.random.binomial(1, 0.8, (n,1))
#	obs = np.hstack((x,y,z)).astype(int)
#	s = SPN(3, SPNParams(mergebatch=1000, leaftype="binary"))
#	p = s.root.children[0]

	s.update(obs)
	s.display()

