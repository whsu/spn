import numpy as np

class Experiment:
	def __init__(self, model, trainfiles, testfiles):
		self.model = model
		self.trainfiles = trainfiles
		self.testfiles = testfiles

	def train_one_file(self, filename):
		dtype = int if self.model.params.binary else float
		obs = np.loadtxt(filename, delimiter=",")
		self.model.update(obs)

	def train(self):
		for filename in self.trainfiles:
			print(filename)
			self.train_one_file(filename)

	def evaluate_one_file(self, filename):
		dtype = int if self.model.params.binary else float
		obs = np.loadtxt(filename, delimiter=",")
		logprob = self.model.evaluate(obs)
		return logprob

	def evaluate(self):
		logprob_total = 0.0
		n_total = 0
		for filename in self.testfiles:
			logprob = self.evaluate_one_file(filename)
			logprob_total += np.sum(logprob)
			n_total += len(logprob)
		return logprob_total / n_total

	def run(self):
		self.train()
		result = self.evaluate()
		return result

