from spn.spn import *
from spn.sum_node import SumNode
from spn.normal_leaf_node import NormalLeafNode
from spn.multi_normal_leaf_node import MultiNormalLeafNode

def count_nodes(network):
	nextnodes = [network.root.children[0]]
	count = 0
	while len(nextnodes) > 0:
		node = nextnodes.pop()
		count += 1
		nextnodes.extend(node.children)
	return count

def count_params(network):
	nextnodes = [network.root.children[0]]
	count = 0
	while len(nextnodes) > 0:
		node = nextnodes.pop()
		if type(node) == SumNode:
			count += len(node.children)
		elif type(node) == NormalLeafNode:
			count += 2
		elif type(node) == MultiNormalLeafNode:
			k = len(node.scope)
			count += k*(k+3)//2
		nextnodes.extend(node.children)
	return count
	
