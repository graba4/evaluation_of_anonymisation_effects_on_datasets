import cli
import ci_decisiontrees


data, model, anonimity, param = cli.parse()
print('parsed')

if data == 'census':
	if model == 'decisiontree':
		#ci_decisiontrees.rest()
		print(param[0])
		print(param[1])
	elif model == 'knn':
		#ci_decisiontrees.rest()
		print(param[0])
	elif model == 'neuralnetwork':
		#ci_decisiontrees.rest()
		print(param[0])
		print(param[1])
		print(param[2])
	else:
		print('no more implemented ml models')
elif data == 'other':
	if model == 'decisiontree':
		#ci_decisiontrees.rest()
		print(param[0])
		print(param[1])
	elif model == 'knn':
		#ci_decisiontrees.rest()
		print(param[0])
	elif model == 'neuralnetwork':
		#ci_decisiontrees.rest()
		print(param[0])
		print(param[1])
		print(param[2])
	else:
		print('no more implemented ml models')
else:
	print('no more datasets available')	

