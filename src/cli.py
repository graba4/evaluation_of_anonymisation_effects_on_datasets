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

	args = parser.parse_args()

	dataset = args.d
	model = args.m

	"""
	if dataset == 'census':
		data_path = '/home/graba4/Desktop/ML_Exercise1_BegovicA_HodovicM_KovacsP/CI/adult.data'
	else:
		# other path
		data_path = '/home/graba4/Desktop/ML_Exercise1_BegovicA_HodovicM_KovacsP/CI/adult.names'

	if model == 'decisiontree':
		model_path = 1
	elif model == 'neuralnetwork':
		model_path = 2
	else:
		model_path = 3
	"""
	return dataset, model

#data1, model1 = parse()

#census_income= pd.read_csv(data1, sep=',', engine='python', na_values='?')
#print(census_income.iloc[0])