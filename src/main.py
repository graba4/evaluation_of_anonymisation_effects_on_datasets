import cli
import ci_decisiontrees
import K_ano_Census
import K_ano_Titan
import census_correlated
import titanic_correlated
import Census_Eval_NN
import Titanic_Eval_NN


data, model, anonimity, param, k_values, do_ano = cli.parse()
print('parsed')

if data == 'census':
	# anonymization part will be done here
	if anonimity == 'gen':
		if do_ano == 'yes':
			for i in range(len(k_values)):
				K_ano_Census.K_ano_Census_func(int(k_values[i]))
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
		# call method that does training and evaluation based on parsed parameters
		Census_Eval_NN.train_and_eval(k_values, param[0], param[1], param[2])
	else:
		print('no more implemented ml models')
elif data == 'titanic':
	# anonymization part will be done here
	if anonimity == 'gen':
		if do_ano == 'yes':
			for i in range(len(k_values)):
				K_ano_Titan.K_ano_Titan_func(int(k_values[i]))
	elif anonimity == 'synth':
		if do_ano == 'yes':
			titanic_correlated.synthetize()
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
		# call method that does training and evaluation based on parsed parameters
		Titanic_Eval_NN.train_and_eval(k_values, param[0], param[1], param[2])
	else:
		print('no more implemented ml models')
else:
	print('no more datasets available')	

