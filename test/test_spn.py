import unittest

from spn.spn import *
from spn.product_node import *
from spn.normal_leaf_node import *
from spn.multi_normal_leaf_node import *
from spn.sum_node import *

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

