"""
This is the main script for the classification task.
"""

import pandas as pd

from classification import select_feature, get_dummies
from classification.models import model_best

"""
Parameters
"""
# working directory
base_path =  ''
feature_file = 'features.csv'

mode = ['baseline', 'absolute', 'clustered']
level = ['event', 'set', 'case'] # 'activity'
tree_min = 1
tree_max = 4
criteria = ['gini', 'entropy']

"""
Read data and select features
"""
feature_data = pd.read_csv(base_path + feature_file, sep=',', header=0)

features = list(feature_data.columns)

features.remove('response')
features.remove('end')
features.remove('start')
features.remove('case')


"""
Train classifier and make predictions
"""
for m in mode:
	# select features
	if m == 'baseline':
		# baseline model
		selected_features = select_feature(features, None, m)

		print('Fit model with the following features: %s (|%s|)' %
			  (selected_features, len(selected_features)))

		X = feature_data[selected_features]

		# set response variable
		Y = list(feature_data['response'])
		for idx, y in enumerate(Y):
			if y == 0:
				Y[idx] = 1
			else:
				Y[idx] = 0

		class_ratio = []
		for c in [0 ,1]:
			class_ratio.append(Y.count(c))

		print('class ratio (not conf/ conf): %s' % class_ratio)

		# make prediction
		model_best(X, Y, tree_min, tree_max, criteria, m, m, base_path)
	else:
		for l in level:
			# select features
			selected_features = select_feature(features, l, m)

			print('Fit model with the following features: %s (|%s|)' %
				  (selected_features, len(selected_features)))

			X = feature_data[selected_features]

			if m == 'clustered':
				selected_features, X = get_dummies(X, l)

			# response variable
			Y = list(feature_data['response'])
			for idx, y in enumerate(Y):
				if y == 0:
					Y[idx] = 1
				else:
					Y[idx] = 0

			class_ratio = []
			for c in [0, 1]:
				class_ratio.append(Y.count(c))
			print('class ratio (not conf/ conf): %s' % class_ratio)

			# make prediction
			model_best(X, Y, tree_min, tree_max, criteria, l, m, base_path)
