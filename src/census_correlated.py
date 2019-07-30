import os, sys
sys.path.append(os.getcwd() + '/DataSynthesizer/')
from DataDescriber import DataDescriber
from DataGenerator import DataGenerator
from ModelInspector import ModelInspector
from lib.utils import read_json_file, display_bayesian_network
import pandas as pd


def synthetize():
	#get_ipython().run_line_magic('matplotlib', 'auto')
	# Adding current direcotry into sys.path
	# input dataset
	input_data = './data/adult_ssn.csv'
	# location of two output files
	mode = 'correlated_attribute_mode'
	description_file = f'./out/{mode}/description.json'
	synthetic_data = f'./out/{mode}/sythetic_data.csv'


	# An attribute is categorical if its domain size is less than this threshold.
	# Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
	threshold_value = 20

	# specify categorical attributes
	categorical_attributes = {'education': True}

	# specify which attributes are candidate keys of input dataset.
	candidate_keys = {'ssn': True}

	# A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not 
	# change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
	# Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
	epsilon = 0.1

	# The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
	degree_of_bayesian_network = 2

	# Number of tuples generated in synthetic dataset.
	num_tuples_to_generate = 32561 # Here 32561 is the same as input dataset, but it can be set to another number.


	# ### Step 3 DataDescriber
	# 
	# 1. Instantiate a DataDescriber.
	# 2. Compute the statistics of the dataset.
	# 3. Save dataset description to a file on local machine.

	describer = DataDescriber(category_threshold=threshold_value)
	describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
	                                                        epsilon=epsilon, 
	                                                        k=degree_of_bayesian_network,
	                                                        attribute_to_is_categorical=categorical_attributes,
	                                                        attribute_to_is_candidate_key=candidate_keys)
	describer.save_dataset_description_to_file(description_file)

	display_bayesian_network(describer.bayesian_network)


	# ### Step 4 generate synthetic dataset
	# 
	# 1. Instantiate a DataGenerator.
	# 2. Generate a synthetic dataset.
	# 3. Save it to local machine.

	generator = DataGenerator()
	generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
	generator.save_synthetic_data(synthetic_data)


	# ### Step 5 compare the statistics of input and sythetic data (optional)
	# 
	# The synthetic data is already saved in a file by step 4. The ModelInspector is for a quick test on the similarity between input and synthetic datasets.
	# 
	# #### 5.1 instantiate a ModelInspector.
	# 
	# It needs input dataset, synthetic dataset, and attribute description.

	# Read both datasets using Pandas.
	input_df = pd.read_csv(input_data, skipinitialspace=True)
	synthetic_df = pd.read_csv(synthetic_data)
	# Read attribute description from the dataset description file.
	attribute_description = read_json_file(description_file)['attribute_description']

	inspector = ModelInspector(input_df, synthetic_df, attribute_description)


	# #### 5.2 compare histograms between input and synthetic datasets.

	plot_id = 0
	for attribute in synthetic_df.columns:
	    inspector.compare_histograms(attribute, plot_id)
	    plot_id += 1


	# #### 5.3 compare pairwise mutual information

	inspector.mutual_information_heatmap()
