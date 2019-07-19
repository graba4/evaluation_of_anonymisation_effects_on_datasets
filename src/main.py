import cli
import ci_decisiontrees


data1, model1 = cli.parse()
print('parsed')

if data1 == 'census' and model1 == 'decisiontree':
	ci_decisiontrees.rest()
else:
	print('not implemented yet')

