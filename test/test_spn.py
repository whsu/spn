import unittest

from spn.spn import *
from spn.product_node import *
from spn.normal_leaf_node import *
from spn.multi_normal_leaf_node import *
from spn.sum_node import *
from spn.multi_normal_leaf_node import *
from spn.normal_leaf_node import *

def normal_test_data(n):
	np.random.seed(0)
	mean = np.array([1., 2., 3.])
	cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
	obs = np.random.multivariate_normal(mean, cov, n)
	return obs, mean, cov

class TestSPN(unittest.TestCase):
	def test_product_evaluate_1(self):
		child1 = NormalLeafNode(10, 0, 0.0, 1.0)
		child2 = NormalLeafNode(10, 0, 1.0, 4.0)
		node = ProductNode(10, np.array([0]), "normal")
		node.children.append(child1)
		node.children.append(child2)
		s = SPN(node, SPNParams())
		np.testing.assert_almost_equal(s.evaluate(np.array([0.0])), -2.656024246969)

	def test_product_sum_evaluate_1(self):
		grandchild1 = NormalLeafNode(3, 0, 0.0, 1.0)
		grandchild2 = NormalLeafNode(7, 0, 1.0, 4.0)
		child1 = SumNode(10, np.array([0]))
		child1.children.append(grandchild1)
		child1.children.append(grandchild2)
		child2 = NormalLeafNode(10, 1, -1.0, 1.0)
		node = ProductNode(10, np.array([0,1]), "normal")
		node.children.append(child1)
		node.children.append(child2)
		s = SPN(node, SPNParams())
		np.testing.assert_almost_equal(s.evaluate(np.array([0.0, 1.0])), -4.3038903197602858)

	def test_product_mv_update_1(self):
		np.random.seed(0)
		mean = np.array([1., 2., 3.])
		cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
		n = 10000
		obs = np.random.multivariate_normal(mean, cov, n)
		child1 = MultiNormalLeafNode.create(0, np.array([0,1]))
		child2 = NormalLeafNode(0, 2, 0.0, 0.0)
		node = ProductNode(0, np.array([0,1,2]), "normal")
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
		# Sum node uses add-one smoothing, so the expected result is
		# log( (4/12)*N(0|0,1)+(8/12)*N(0|1,4) ) = -1.3849517865556134
		np.testing.assert_almost_equal(s.evaluate(np.array([0.0])), -1.38495179)

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

	def test_multi_normal_leaf_node_update(self):
		n = 100000
		m = 100
		obs, mean, cov = normal_test_data(n)
		node = MultiNormalLeafNode.create(0, np.arange(3))
		params = SPNParams()
		for k in range(0, n, m):
			node.update(obs[k:k+m,:], params)
		self.assertEqual(node.n, n)
		np.testing.assert_almost_equal(node.stat.mean, mean, decimal=2)
		np.testing.assert_almost_equal(node.stat.cov, cov, decimal=2)

	def test_normal_leaf_node_update(self):
		n = 300000
		m = 100
		obs, mean, cov = normal_test_data(n)
		nodes = [NormalLeafNode(0, i) for i in range(3)]
		params = SPNParams()
		for k in range(0, n, m):
			for i in range(3):
				nodes[i].update(obs[k:k+m,:], params)
		self.assertEqual(nodes[0].n, n)
		self.assertEqual(nodes[1].n, n)
		self.assertEqual(nodes[2].n, n)
		self.assertAlmostEqual(nodes[0].mean, mean[0], 2)
		self.assertAlmostEqual(nodes[1].mean, mean[1], 2)
		self.assertAlmostEqual(nodes[2].mean, mean[2], 2)
		self.assertAlmostEqual(nodes[0].var, cov[0,0], 2)
		self.assertAlmostEqual(nodes[1].var, cov[1,1], 2)
		self.assertAlmostEqual(nodes[2].var, cov[2,2], 2)

