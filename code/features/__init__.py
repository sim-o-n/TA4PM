"""
This file contains functions that are used in multiple scripts of this
package. It contains functions to:
 - import conformance reports
 - generate aggregation names
 - perform LOF and k-means clustering
 - visualize data
"""

import os.path
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pathlib
from sklearn.cluster import KMeans, DBSCAN

from util import print_time


def import_report(file: str):
	"""
	This function is used to import a conformance report.

	:param file: conformance report
	:return: pandas DataFrame, containing raw cost per case
	"""

	print_time('Import report')
	df = pd.read_csv(file, sep=',', header=[0])

	costs = []
	cases = []

	for idx, row in enumerate(df.loc[:, 'Case IDs']):

		raw_cost = df.loc[idx, 'Raw Fitness Cost']

		# update alignment id in case table
		for case in df.loc[idx, 'Case IDs'].split('|'):
			cases.append(case)
			costs.append(float(raw_cost))

	print_time('Import report', False)

	df = pd.DataFrame({'case': cases, 'response': costs})
	return df


def get_aggregation_attributes(columns: list, entities: list, prefix=None):
	"""
	This function can be used to generate a list of aggregation names.

	:param columns:  attributes
	:param entities: observations
	:param prefix: optional prefix
	:return: list of name
	"""

	aggr = []
	for c in columns:
		for e in entities:
			if prefix is None:
				aggr.append('_'.join([str(c), str(e)]))
			else:
				aggr.append('_'.join([prefix, str(c), str(e)]))

	return aggr


def do_LOF(data: pd.DataFrame, column_name: str, base_path: str,
		   neighbors: int = 100, plot: bool = False):
	"""
	This function performs a LOF computation for given data.

	:param data: features
	:param column_name: feature name
	:param base_path: path to folder, to save plots
	:param neighbors: number of neighbors to consider
	:param plot: whether to visualize the LOF or not
	"""

	# select column
	data[column_name + '_' + 'outlier'] = [None] * len(data)

	# configure algorithm
	algorithm = LocalOutlierFactor(n_neighbors=neighbors,
								   contamination=0.2)

	# prepare data
	X = np.array(data[column_name]).reshape(-1, 1)

	# fit LOF and assign classes to the observations
	algorithm.fit(X)
	y_pred = algorithm.fit_predict(X)

	for idx, y in enumerate(y_pred):
		data.at[idx, column_name + '_' + 'outlier'] = y == - 1

	if plot:
		histo_scatter(data, X, y_pred, column_name, base_path)


def clusters_to_csv(data: dict, file: str):
	"""
	Save cluster ranges to CSV. If the given CSV file already exist,
	the results will be appended.

	:param data: cluster ranges
	:param file: path to file to save findings
	"""

	# generate DataFrame
	df = pd.DataFrame(data)

	if os.path.isfile(file):
		# append to file
		df.to_csv(file, mode='a', index=False, header=False)
	else:
		# make new file
		df.to_csv(file, index=False, header=True)


def do_kmeans(data: pd.DataFrame, X_name: str, path: str, plot: bool = False,
			  k: int = 3):
	"""
	This function performs a k-means clustering for the given data.
	WARNING: for large number of observations turn off the visualization,
	since due to high computational effort.

	:param data: features
	:param X_name: name of the feature to cluster
	:param path: location to save results
	:param plot: whether to visualize k-means result
	:param k: number of clusters
	"""

	# set cluster names
	clusters = {}
	for c in range(0, k):
		cluster_name = '_'.join(['cluster', str(c)])
		clusters[cluster_name] = []

	# prepare feature DataFrame
	data[X_name + '_' + 'cluster'] = [None] * len(data)

	# adjust number of clusters if not sufficient unique values exist.
	unique_v = len(list(data[X_name].unique()))
	if unique_v < 3:
		k = unique_v

	X = np.array(data[X_name]).reshape(-1, 1)

	# find clusters and assign memberships to observations
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	X_pred = kmeans.predict(X)

	# set cluster memberships
	values = list(data[X_name])
	for idx, cluster in enumerate(X_pred):
		value = values[idx]
		data.at[idx, X_name + '_' + 'cluster'] = cluster
		clusters['_'.join(['cluster', str(cluster)])].append(value)

	# get cluster ranges
	for k, v in clusters.items():
		if len(v) > 0:
			l_bound = min(v)
			u_bound = max(v)
			clusters[k] = ['%i-%i' % (l_bound, u_bound)]
		else:
			clusters[k] = [None]

	clusters['feature'] = X_name

	# save cluster ranges
	clusters_to_csv(clusters, 'cluster_info.csv')

	if plot:
		# visualize clusters and data distribution

		fig, (a0, a1) = plt.subplots(2, 1,
									 gridspec_kw={'height_ratios': [4, 1]})
		fig.suptitle(X_name)
		colors = np.array(['r', 'b', 'g', 'c', 'm', 'y', 'k'])

		a0.hist(X, bins='auto', color='b')
		a0.set_ylabel('number of events')

		a1.scatter(X[:, 0], [0] * len(X), s=5, color=colors[X_pred],
				   alpha=0.5)

		a1.set_xlabel('time in seconds')
		a1.set_yticklabels([])

		# save plot as pdf
		name = ''.join([X_name, '_histo_scatter', '.pdf'])
		path = path + '/kmeans'
		pathlib.Path(path).mkdir(parents=True, exist_ok=True)
		path = '/'.join([path, name])

		fig.savefig(path, bbox_inches='tight')
		plt.clf()
		plt.close(fig)


def histo_scatter(data: pd.DataFrame, X: list, y_pred: list, column_name: str,
				  path: str):
	"""
	This function plots a histogram of the data distribution and the
	membership of each single observation with respect to LOF. The
	generated plot will be saved as PDF.

	:param data: feature set
	:param X: feature values
	:param y_pred: membership
	:param column_name: feature name
	:param path: path to save plot
	"""
	fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
	fig.suptitle(column_name)
	colors = np.array(['r', 'b'])

	# histogram
	a0.hist(list(data[column_name]), bins='auto', color=colors[1])
	a0.set_ylabel('number of events')

	# memberships
	a1.scatter(X[:, 0], [0] * len(X), s=5, color=colors[(y_pred + 1) // 2],
			   alpha=0.5)
	a1.set_xlabel('time in seconds')
	a1.set_yticklabels([])

	# export
	name = ''.join([column_name, '_histo_scatter', '.pdf'])
	path = path + '/LOF'
	pathlib.Path(path).mkdir(parents=True, exist_ok=True)
	path = '/'.join([path, name])

	fig.savefig(path, bbox_inches='tight')
	plt.clf()
	plt.close(fig)
