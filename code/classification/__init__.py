"""
This file contains functions, that are used for feature selection and
one-encoding.
"""

import pandas as pd


def select_feature(feature: list, level: list, feature_type: list):
	"""
	Get all features that refer to the given process granularity and scenario.

	:param feature: all features
	:param level: process granularity
	:param feature_type: scenario
	:return: selected features as list
	"""

	selected_features = []

	if level is None and feature_type == 'baseline':
		for f in feature:
			if 'involved' in f:
				selected_features.append(f)

	if 'event' == level:
		for f in feature:
			if 'case' not in f and 'set' not in f and 'act' not in f and \
					'involved' not in f:
				if 'clustered' in feature_type and 'cluster' in f:
					selected_features.append(f)

				if 'absolute' in feature_type and 'cluster' not in f:
					selected_features.append(f)

	if 'set' == level:
		for f in feature:
			if 'set' in f:
				if 'clustered' in feature_type and 'cluster' in f:
					selected_features.append(f)

				if 'absolute' in feature_type and 'cluster' not in f:
					selected_features.append(f)

	if 'activity' == level:
		for f in feature:
			if 'act' in f:
				if 'clustered' in feature_type and 'cluster' in f:
					selected_features.append(f)

				if 'absolute' in feature_type and 'cluster' not in f:
					selected_features.append(f)

	if 'case' == level:
		for f in feature:
			if 'case' in f:
				if 'clustered' in feature_type and 'cluster' in f:
					selected_features.append(f)

				if 'absolute' in feature_type and 'cluster' not in f:
					selected_features.append(f)

	return selected_features


def get_dummies(X: pd.DataFrame, level: str):
	"""
	Generate dummy variables (one-hot encoding) for given features.

	:param X: feature data
	:param level: process granularity
	:return: selected features as list,
			feature data
	"""

	if 'event' == level:
		for c in X.columns:
			X[c] = X[c].astype('category')
		# one-hot encode categorical data
		X = pd.get_dummies(X)
		selected_features = X.columns

	if 'set' == level:
		for c in X.columns:
			if 'cluster' in c or 'start' in c or 'weekday' in c and \
					'avg' not in c:
				X[c] = X[c].astype('category')
		X = pd.get_dummies(X)
		selected_features = X.columns
		print(X.columns)

	if 'activity' == level:
		for c in X.columns:
			if 'act' in c or 'start' in c and 'avg' not in c:
				X[c] = X[c].astype('category')
		X = pd.get_dummies(X)
		selected_features = X.columns
		print(X.columns)

	if 'case' == level:
		# X = X.drop('case_duration', axis=1)
		for c in X.columns:
			if 'cluster' in c or 'start' in c and 'avg' not in c:
				X[c] = X[c].astype('category')
		X = pd.get_dummies(X)
		selected_features = X.columns
		print(X.columns)

	return selected_features, X
