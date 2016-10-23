import os
import time
import pickle

import numpy as np

from spn.spn import *
from .experiment import Experiment

DATADIR = 'data'
OUTDIR = 'output'

def make_kfold_filenames(vartype, name, k):
	filenames = [os.path.join(DATADIR, vartype, name, "{0}.{1}.data".format(name, i+1))
	             for i in range(k)]
	return filenames

def make_train_test_filenames(vartype, name):
	trainfiles = [os.path.join(DATADIR, vartype, name, "{0}.train.data".format(name))]
	testfiles = [os.path.join(DATADIR, vartype, name, "{0}.test.data".format(name))]
	return trainfiles, testfiles

def run_train_test(trainfiles, testfiles, numvar, params):
	model = SPN(numvar, params)
	experiment = Experiment(model, trainfiles, testfiles)
	t0 = time.clock()
	result = experiment.run()
	t1 = time.clock()
	return result, t1-t0, model

def run_ith_fold(i, filenames, numvar, params):
	testfiles = filenames[i:i+1]
	trainfiles = filenames[i+1:] + filenames[:i]
	return run_train_test(trainfiles, testfiles, numvar, params)

def run_kfold(vartype, name, k, numvar, params):
	filenames = make_kfold_filenames(vartype, name, k)
	results = [None] * k
	times = [None] * k
	models = [None] * k
	for i in range(k):
		results[i], times[i], models[i] = run_ith_fold(i, filenames, numvar, params)
		print(i, results[i], times[i])
	print(np.mean(results), np.std(results))
	return results, times, models

def run(vartype, traintest, name, numvar, mergebatch, corrthresh, equalweight, updatestruct,
	    maxdepth, prunebatch, mvleaf, mvmaxscope, binary):
	outfile = "{0}_{1}_{2}_{3}_{4}_{5}".format(name, mergebatch,
                           corrthresh, maxdepth, prunebatch, mvmaxscope)
	resultpath = os.path.join(OUTDIR, "{0}.txt".format(outfile))
	picklepath = os.path.join(OUTDIR, "{0}.pkl".format(outfile))
	params = SPNParams(mergebatch, corrthresh, equalweight, updatestruct,
	    maxdepth, prunebatch, mvleaf, mvmaxscope, binary)

	print('******{0}*******'.format(name))

	if traintest:
		trainfiles, testfiles = make_train_test_filenames(vartype, name)
		result, t, model = run_train_test(trainfiles, testfiles, numvar, params)
		print('Loglhd: {0:.3f}\n'.format(result))
		print('Time: {0:.3f}\n'.format(t))
		with open(resultpath, 'w') as g:
			g.write('Loglhd: {0:.3f}\n'.format(result))
			g.write('Time: {0:.3f}\n'.format(t))
			g.close()
		with open(picklepath, 'wb') as g:
			pickle.dump(model, g)
		print(model)
	else:
		results, times, models = run_kfold(vartype, name, 10, numvar, params)

		with open(resultpath, 'w') as g:
			g.write('Loglhd: {0:.3f}, {1:.3f}\n'.format(np.mean(results), np.std(results)))
			g.write('Times: {0:.3f}, {1:.3f}\n'.format(np.mean(times), np.std(times)))
			g.write('\n')
			for r, t in zip(results, times):
				g.write('{0:.3f} {1:.3f}\n'.format(r, t))
			g.close()
		with open(picklepath, 'wb') as g:
			pickle.dump(models, g)
		print(models)

if __name__ == '__main__':
#	run('real', False, 'qu', 4, 10, 0.1, True, True, 100, 1000, True, 4, False)
#	run('real', False, 'banknote', 4, 10, 0.1, True, True, 100, 1000, True, 4, False)
#	run('real', False, 'sensorless_drive', 48, 1000, 0.1, True, True, 100, 1000, True, 4, False)
#	run('real', False, 'flowdata', 3, 10000, 0.1, True, True, 100, 1000, True, 4, False)
#	run('real', False, 'abalone', 8, 10, 0.1, True, True, 100, 1000, True, 4, False)
#	run('real', False, 'ki', 8, 10, 0.1, True, True, 100, 1000, True, 4, False)
#	run('real', False, 'ca', 22, 10, 0.1, True, True, 100, 1000, True, 4, False)

	run('binary', True, 'nltcs', 16, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'plants', 69, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'baudio', 100, 100, 0.1, True, True, 50, 1000, False, 4, True)
	run('binary', True, 'jester', 100, 10, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'bnetflix', 100, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'accidents', 111, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'tretail', 135, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'pumsb_star', 163, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'dna', 180, 10, 0.1, True, True, 100, 50, False, 4, True)
#	run('binary', True, 'kosarek', 190, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'msweb', 294, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'book', 500, 10, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'tmovie', 500, 10, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'cwebkb', 839, 10, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'cr52', 889, 10, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'c20ng', 910, 100, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'bbc', 1058, 10, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'ad', 1556, 10, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'msnbc', 17, 10000, 0.1, True, True, 50, 1000, False, 4, True)
#	run('binary', True, 'kdd', 64, 10000, 0.1, True, True, 50, 1000, False, 4, True)

