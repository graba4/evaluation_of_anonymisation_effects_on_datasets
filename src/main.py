import cli
import ci_decisiontrees
import census_k_anonimdata
import census_correlated


data, model, anonimity, param, k_values, do_ano = cli.parse()
print('parsed')

if data == 'census':
	# anonymization part will be done here
	if anonimity == 'gen':
		if do_ano == 'yes':
			for i in range(len(k_values)):
				census_k_anonimdata.build_dataset_of_k_anonymity(int(k_values[i]))
	elif anonimity == 'synth':
		if do_ano == 'yes':
			census_correlated.synthetize()
	else:
		print('no more anonymization methods implemented')

	########################################
	if model == 'decisiontree':
		#ci_decisiontrees.rest()
		# call method that does training and evaluation based on parsed parameters
		print(param[0])
		print(param[1])
	elif model == 'knn':
		#ci_decisiontrees.rest()
		# call method that does training and evaluation based on parsed parameters
		print(param[0])
	elif model == 'neuralnetwork':
		#ci_decisiontrees.rest()
		# call method that does training and evaluation based on parsed parameters
		print(param[0])
		print(param[1])
		print(param[2])
	else:
		print('no more implemented ml models')
elif data == 'titanic':
	# anonymization part will be done here
	if anonimity == 'gen':
		# peters k-ano .py
		print('not yet')
	elif anonimity == 'synth':
		# amars datasynth .py
		print('not yet')
	else:
		print('no more anonymization methods implemented')

	########################################
	if model == 'decisiontree':
		#ci_decisiontrees.rest()
		# call method that does training and evaluation based on parsed parameters
		print(param[0])
		print(param[1])
	elif model == 'knn':
		#ci_decisiontrees.rest()
		# call method that does training and evaluation based on parsed parameters
		print(param[0])
	elif model == 'neuralnetwork':
		#ci_decisiontrees.rest()
		# call method that does training and evaluation based on parsed parameters
		print(param[0])
		print(param[1])
		print(param[2])
	else:
		print('no more implemented ml models')
else:
	print('no more datasets available')	

