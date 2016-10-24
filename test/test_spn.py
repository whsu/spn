import unittest

from spn.spn import *
from spn.product_node import *
from spn.normal_leaf_node import *
from spn.multi_normal_leaf_node import *
from spn.sum_node import *
from spn.binary_leaf_node import *
from spn.binary_normal_leaf_node import *
from spn.multi_binary_leaf_node import *
from spn.multi_binary_normal_leaf_node import *
from spn.multi_normal_leaf_node import *
from spn.normal_leaf_node import *

def normal_test_data(n):
	np.random.seed(0)
	mean = np.array([1., 2., 3.])
	cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
	obs = np.random.multivariate_normal(mean, cov, n)
	return obs, mean, cov

def binary_test_data(n):
	'''
	P(x0=1) = 0.1
	P(x1=1|x0=0) = 0.2
	P(x1=1|x0=1) = 0.9
	P(x1=1) = 0.27
	P(x2=1) = 0.8
	'''
	np.random.seed(0)
	x0 = np.random.binomial(1, 0.1, (n,1))
	x1 = np.empty((n,1))
	i = np.where(x0==0)[0]
	j = np.where(x0==1)[0]
	x1[i] = np.random.binomial(1, 0.2, (len(i),1))
	x1[j] = np.random.binomial(1, 0.9, (len(j),1))
	x2 = np.random.binomial(1, 0.8, (n,1))
	obs = np.hstack((x0,x1,x2)).astype(int)

	# expected marginal probabilities
	marg = np.array([0.1, 0.27, 0.8])

	# expected pairwise marginal probabilities
	margpair = {(0,1): np.array([[0.72 ,0.18 ],[0.01 ,0.09 ]]),
	            (0,2): np.array([[0.18 ,0.72 ],[0.02 ,0.08 ]]),
	            (1,2): np.array([[0.146,0.584],[0.054,0.216]])}

	# expected covariance matrix
	cov = np.array([[0.09 , 0.063 , 0.0],
	                [0.063, 0.1971, 0.0],
	                [0.0  , 0.0   , 0.16]])

	return obs, marg, margpair, cov

class TestSPN(unittest.TestCase):
	def test_product_evaluate_1(self):
		child1 = NormalLeafNode(10, 0, 0.0, 1.0)
		child2 = NormalLeafNode(10, 0, 1.0, 4.0)
		node = ProductNode(10, np.array([0]))
		node.children.append(child1)
		node.children.append(child2)
		s = SPN(node, SPNParams())
		self.assertAlmostEqual(s.evaluate(np.array([0.0])), -2.656024246969)

	def test_product_sum_evaluate_1(self):
		grandchild1 = NormalLeafNode(3, 0, 0.0, 1.0)
		grandchild2 = NormalLeafNode(7, 0, 1.0, 4.0)
		child1 = SumNode(10, np.array([0]))
		child1.children.append(grandchild1)
		child1.children.append(grandchild2)
		child2 = NormalLeafNode(10, 1, -1.0, 1.0)
		node = ProductNode(10, np.array([0,1]))
		node.children.append(child1)
		node.children.append(child2)
		s = SPN(node, SPNParams())
		self.assertAlmostEqual(s.evaluate(np.array([0.0, 1.0])), -4.33402113)

	def test_product_mv_update_1(self):
		np.random.seed(0)
		mean = np.array([1., 2., 3.])
		cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
		n = 10000
		obs = np.random.multivariate_normal(mean, cov, n)
		child1 = MultiNormalLeafNode.create(0, np.array([0,1]))
		child2 = NormalLeafNode(0, 2, 0.0, 0.0)
		node = ProductNode(0, np.array([0,1,2]))
		node.children = [child1, child2]
		s = SPN(node, SPNParams())
		for x in obs:
			s.update(x)
		self.assertEqual(node.n, n)
		self.assertEqual(child1.n, n)
		self.assertEqual(child2.n, n)
		np.testing.assert_almost_equal(child1.stat.mean, mean[:2], decimal=1)
		np.testing.assert_almost_equal(child1.stat.cov, cov[np.ix_([0,1],[0,1])], decimal=1)
		np.testing.assert_almost_equal(child2.mean, mean[2], decimal=1)
		np.testing.assert_almost_equal(child2.var, cov[2,2], decimal=1)

	def test_sum_evaluate_1(self):
		child1 = NormalLeafNode(3, 0, 0.0, 1.0)
		child2 = NormalLeafNode(7, 0, 1.0, 4.0)
		node = SumNode(10, np.array([0]))
		node.children.append(child1)
		node.children.append(child2)
		s = SPN(node, SPNParams())
		self.assertAlmostEqual(s.evaluate(np.array([0.0])), -1.41508260055)

	def test_sum_update_1(self):
		child1 = NormalLeafNode(3, 0, 0.0, 1.0)
		child2 = NormalLeafNode(7, 0, 1.0, 4.0)
		node = SumNode(10, np.array([0]))
		node.children = [child1, child2]
		s = SPN(node, SPNParams())
		s.update(np.array([0.0]))

		# equalWeight is true, so update passes the data point to the component
		# with highest likelihood without considering the weight of each component.
		# In this case, N(0|0,1) > N(0|1,4), so child1 is picked.
		# If component weights are taken into account, then child2 will be picked
		# since 0.3*N(0|0,1) < 0.7*N(0|1,4).
		self.assertEqual(node.n, 11)
		self.assertEqual(child1.n, 4)
		self.assertEqual(child2.n, 7)

	def test_binary_leaf_update(self):
		obs, marg, margpair, cov = binary_test_data(100000)
		nodes = [BinaryLeafNode(0, i, 0) for i in range(3)]
		params = SPNParams()
		for x in obs:
			for i in range(3):
				nodes[i].update(x, params)
		self.assertAlmostEqual(nodes[0].p, marg[0], 2)
		self.assertAlmostEqual(nodes[1].p, marg[1], 2)
		self.assertAlmostEqual(nodes[2].p, marg[2], 2)

	def test_binary_normal_leaf_update(self):
		obs, marg, margpair, cov = binary_test_data(100000)
		nodes = [BinaryNormalLeafNode(0, i, 0) for i in range(3)]
		params = SPNParams()
		for x in obs:
			for i in range(3):
				nodes[i].update(x, params)
		self.assertAlmostEqual(nodes[0].p, marg[0], 2)
		self.assertAlmostEqual(nodes[1].p, marg[1], 2)
		self.assertAlmostEqual(nodes[2].p, marg[2], 2)

	def test_multi_binary_leaf_update(self):
		obs, marg, margpair, cov = binary_test_data(100000)
		node = MultiBinaryLeafNode.create(0, np.arange(3))
		params = SPNParams()
		for x in obs:
			node.update(x, params)
		np.testing.assert_almost_equal(node.stat.probs, marg, decimal=2)
		self.assertEqual(set(node.stat.matrices.keys()),
		                 set([(0,1),(0,2),(1,2)]))
		np.testing.assert_almost_equal(node.stat.matrices[(0,1)],
		                               margpair[(0,1)], decimal=2)
		np.testing.assert_almost_equal(node.stat.matrices[(0,2)],
		                               margpair[(0,2)], decimal=2)
		np.testing.assert_almost_equal(node.stat.matrices[(1,2)],
		                               margpair[(1,2)], decimal=2)

	def test_multi_binary_normal_leaf_update(self):
		obs, marg, margpair, cov = binary_test_data(100000)
		node = MultiBinaryNormalLeafNode.create(0, np.arange(3))
		params = SPNParams()
		for x in obs:
			node.update(x, params)
		np.testing.assert_almost_equal(node.stat.normal.mean, marg, decimal=2)
		np.testing.assert_almost_equal(node.stat.normal.cov, cov, decimal=2)

	def test_multi_normal_leaf_node_update(self):
		obs, mean, cov = normal_test_data(100000)
		node = MultiNormalLeafNode.create(0, np.arange(3))
		params = SPNParams()
		for x in obs:
			node.update(x, params)
		np.testing.assert_almost_equal(node.stat.mean, mean, decimal=2)
		np.testing.assert_almost_equal(node.stat.cov, cov, decimal=2)

	def test_normal_leaf_node_update(self):
		obs, mean, cov = normal_test_data(300000)
		nodes = [NormalLeafNode(0, i) for i in range(3)]
		params = SPNParams()
		for x in obs:
			for i in range(3):
				nodes[i].update(x, params)
		self.assertAlmostEqual(nodes[0].mean, mean[0], 2)
		self.assertAlmostEqual(nodes[1].mean, mean[1], 2)
		self.assertAlmostEqual(nodes[2].mean, mean[2], 2)
		self.assertAlmostEqual(nodes[0].var, cov[0,0], 2)
		self.assertAlmostEqual(nodes[1].var, cov[1,1], 2)
		self.assertAlmostEqual(nodes[2].var, cov[2,2], 2)

