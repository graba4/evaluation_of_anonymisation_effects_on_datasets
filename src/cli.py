import argparse
import pandas as pd


def parse():
	parser = argparse.ArgumentParser()

	parser.add_argument('-d',
						type=str,
						default='census',
						required=True,
						help='defines dataset name',
						choices=['census', 'other'])
	parser.add_argument('-m',
						type=str,
						default='decisiontree',
						required=True,
						help='defines machine learning model',
						choices=['decisiontree', 'neuralnetwork', 'knn'])
	parser.add_argument('-a',
						type=str,
						default='no',
						help='defines whether set is anonymized or not',
						choices=['no', 'yes'])
	parser.add_argument('-pdt',
						type=str,
						nargs=2,
						default=[4, 16],
                    	help='training parameters for decision tree model: MAX_DEPTH, MIN_LEAVES')
	parser.add_argument('-pkn',
						type=str,
						nargs=1,
						default=[4],
                    	help='training parameters for knn model: K')
	parser.add_argument('-pnn',
						type=str,
						default=[1, 16, 50],
						nargs=3,
                    	help='training parameters for neural network model: # of hidden layers, size or layers, # of epochs')
	parser.add_argument('-ak',
						type=str,
						nargs=1,
                    	help='defines k value for k-anonymity')

	args = parser.parse_args()

	dataset = args.d
	model = args.m
	anonymized = args.a

	if model == 'decisiontree':
		parameters = args.pdt
	elif model == 'knn':
		parameters = args.pkn
	elif model == 'neuralnetwork':
		parameters = args.pnn
	else:
		parameters = 0

	return dataset, model, anonymized, parameters
