from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt
from subprocess import check_call
import pandas as pd


def plot_feature_importance(features:list, scores:list, feature_level:str,
							feature_type:str, base_path:str):
	"""
	This function visualizes the feature imporance of all features of an
	decision tree classifier as a bar chart.

	:param features: feature names
	:param scores: feature importance
	:param feature_level: process granularity
	:param feature_type: scenario
	:param base_path: path to export results
	:return:
	"""

	data = pd.DataFrame({'feature': features, 'importance': scores})

	data = data.sort_values(by='importance', ascending=False)
	data = data.reset_index(drop=True)

	# keep only features with importance > 0
	a = 0
	i = 0
	for idx, row in data.iterrows():
		a += row['importance']
		if a > 0.99:
			i = idx + 1
			print('i: %s' %i)
			break

	sub_data = data.iloc[0:i]
	sub_data = sub_data.sort_values(by='importance', ascending=False)

	print(sub_data)
	sub_data.set_index("feature", drop=True, inplace=True)
	ax = sub_data.importance.plot(kind='barh', facecolor='#1F77B4')
	ax.spines['bottom'].set_color('#000000')
	ax.spines['bottom'].set_linewidth(1)
	ax.spines['left'].set_color('#000000')
	ax.spines['left'].set_linewidth(1)
	output_file = ''.join([base_path, 'feature_importance_',
						   feature_level, '_level_',
						   feature_type, '.pdf'])
	ax.xaxis.grid(alpha=0.3)
	ax.xaxis.set_label_text('Importance')
	ax.yaxis.label.set_visible(False)
	plt.box(on=None)
	plt.savefig(output_file, format='pdf', bbox_inches='tight')
	plt.clf()

def print_tree(model, selected_features:list, base_path: str, file_name: str):
	"""
	This function visualizes and exports a given decision tree.

	:param model: decision tree model
	:param selected_features: feature names
	:param base_path: path to export the model
	:param file_name: file name
	"""

	dot_file_name = ''.join(
		[base_path, file_name, '_dt', '.dot'])

	print('clf.classes_', model.classes_)

	export_graphviz(model, out_file=dot_file_name,
					filled=True, rounded=True,
					special_characters=True,
					class_names = ['0', '1'],
					feature_names=selected_features)

	output_file = ''.join([dot_file_name.split('.dot')[0], '.pdf'])

	check_call(['dot','-Tpdf', dot_file_name ,'-o', output_file])

	check_call(['rm', dot_file_name])