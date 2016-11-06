import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from spn.spn import *

def gen_data(n):
	np.random.seed(0)
	mean = np.array([1., 2., 3.])
	cov = np.array([[1.0, 0.5, 0.0],[0.5,2.0,0.0],[0.0,0.0,3.0]])
	obs = np.random.multivariate_normal(mean, cov, n)
	return obs

def run(n, figpos, xlabel, ylabel):
	obs = gen_data(n)

	s = SPN(3, SPNParams(mvmaxscope=0))
	s.update(obs)
	s.display()

	ells = []
	for x in s.root.children[0].children[1].children:
		if x.children[0].index == 0:
			c2, c1 = x.children
		else:
			c1, c2 = x.children
		ell = Ellipse(xy=[c1.mean, c2.mean], width=np.sqrt(c1.var),
		              height=np.sqrt(c2.var), fill=False, linewidth=3)
		ells.append(ell)

	fig = plt.figure(0)
	ax = fig.add_subplot(figpos, aspect='equal')
	ax.plot(obs[:,1], obs[:,0], '.')
	for e in ells:
		ax.add_artist(e)
	ax.set_xlim(-3, 6)
	ax.set_ylim(-2, 4)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

if __name__ == '__main__':
	np.set_printoptions(precision=3)

	run(200, 121, "$x_2$", "$x_1$")
	run(300, 122, "$x_2$", "")

	plt.show()

