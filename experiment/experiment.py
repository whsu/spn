import numpy as np

class Experiment:
	def __init__(self, model, trainfiles, testfiles):
		self.model = model
		self.trainfiles = trainfiles
		self.testfiles = testfiles

	def train_one_file(self, filename):
		dtype = int if self.model.params.binary else float
		f = open(filename)
		i = 0
		for line in f:
			obs = np.fromstring(line, sep=",", dtype=dtype)
			self.model.update(obs)
			i += 1
			if i % self.model.params.mergebatch == 0:
				print(i)
		f.close()

	def train(self):
		for filename in self.trainfiles:
			print(filename)
			self.train_one_file(filename)

	def evaluate_one_file(self, filename):
		dtype = int if self.model.params.binary else float
		logprob = 0.0
		n = 0
		f = open(filename)
		for line in f:
			obs = np.fromstring(line, sep=",", dtype=dtype)
			logprob += self.model.evaluate(obs)
			n += 1
		f.close()
		return logprob, n

	def evaluate(self):
		logprob_total = 0.0
		n_total = 0
		for filename in self.testfiles:
			logprob, n = self.evaluate_one_file(filename)
			logprob_total += logprob
			n_total += n
		return logprob_total / n_total

	def run(self):
		self.train()
		result = self.evaluate()
		return result

