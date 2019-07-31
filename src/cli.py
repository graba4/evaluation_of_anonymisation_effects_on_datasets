import argparse
import pandas as pd


def parse():
	parser = argparse.ArgumentParser()

	parser.add_argument('-d',
						type=str,
						default='census',
						required=True,
						help='defines dataset name',
						choices=['census', 'titanic'])
	parser.add_argument('-m',
						type=str,
						default='decisiontree',
						required=True,
						help='defines machine learning model',
						choices=['decisiontree', 'neuralnetwork', 'knn'])
	parser.add_argument('-a',
						type=str,
						default='gen',
						help='defines which anonymization method is performed on set',
						choices=['gen', 'synth'])
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
						default=[2, 32, 50],
						nargs=3,
                    	help='training parameters for neural network model: # of hidden layers, size or layers, # of epochs')
	parser.add_argument('-ak',
						type=str,
						nargs=6,
						default=[2, 4, 8, 16, 32, 64],
						#required=True,
                    	help='defines array of k values for k-anonymity')
	parser.add_argument('-da',
						type=str,
						default='no',
						help='defines whether anonymization is performed on set in program or is already pre-anonymized set used',
						choices=['no', 'yes'])

	args = parser.parse_args()

	dataset = args.d
	model = args.m
	anonymization = args.a
	k_values = args.ak
	do_ano = args.da

	if model == 'decisiontree':
		parameters = args.pdt
	elif model == 'knn':
		parameters = args.pkn
	elif model == 'neuralnetwork':
		parameters = args.pnn
	else:
		parameters = args.pdt # by default decisition tree model is trained

	return dataset, model, anonymization, parameters, k_values, do_ano
