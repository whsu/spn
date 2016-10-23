
class Node:
	def __init__(self, n, scope):
		self.n = n
		self.scope = scope.copy()
		self.children = []
		self.parent = None

	def add_children(self, *children):
		for child in children:
			self.add_child(child)

	def remove_children(self, *children):
		for child in children:
			self.remove_child(child)

