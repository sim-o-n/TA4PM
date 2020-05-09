from sklearn import metrics
from classification.viz import plot_feature_importance, print_tree
from sklearn.model_selection import train_test_split, cross_val_score, \
	GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np

def model_best(X, y, min_depth, max_depth, crits, level, mode, path):
	"""
	Learn decision tree classifier for different parameter settings. The
	results will be exported and the best performing model will be visualized.

	:param X: features
	:param y: response variable
	:param min_depth: smallest tree size
	:param max_depth: largest tree size
	:param crits: split criteria
	:param level: process granularity
	:param mode: scenario
	:param path: path to export results
	"""

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

	params = {'max_depth': range(min_depth, max_depth + 1),
				  'criterion': crits}
	gs_clf = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-11, cv=5,
						  scoring='f1')
	gs_clf = gs_clf.fit(X_train, y_train)
	model = gs_clf.best_estimator_

	# get scores for all models and save as CSV
	scores_mean = gs_clf.cv_results_['mean_test_score']
	scores_mean = np.array(scores_mean).reshape(len(crits),
												(max_depth + 1) - min_depth)
	scores_mean = [['%s %s (Gini)' %(level, mode)]+list(scores_mean[0]),
				   ['%s %s (entropy)' %(level, mode)] + list(scores_mean[1])]
	df_scores = pd.DataFrame(scores_mean)
	df_scores.to_csv(path + 'scores_mean_f1.csv', index=False, header=False,
					 mode='a')

	# get scores
	acc_tr = model.score(X_train, y_train)
	acc_te = model.score(X_test, y_test)
	print('Accuracy training set %s' %acc_tr)
	print('Accuracy test set %s' %acc_te)

	# Use best model and test data for final evaluation
	y_pred = model.predict(X_test)

	# compute metrics and save to CSV
	tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
	tpr = tp/(tp+fn)
	fpr = fp/(fp+tn)
	ppv = tp/(tp+fp)
	f1 = 2*((ppv * tpr)/(ppv + tpr))

	print('tn: %s,fp: %s,fn: %s,tp: %s,' %(tn, fp, fn, tp))
	print('trp: %s, fpr: %s, ppv: %s, f1: %s' %(tpr,fpr,ppv,f1))

	report = [['%s %s' %(level, mode)]+ [tn, fp, fn, tp, acc_tr, acc_te, tpr,
									   fpr, ppv,
							  f1, model.max_depth]]
	pd.DataFrame(report).to_csv(path + 'report.csv', index=False,
								header=False, mode='a')

	# get feature importance
	feature_imp = model.feature_importances_
	plot_feature_importance(X_train.columns, list(feature_imp), level, mode,
							   path)

	# print process tree
	print_tree(model, X_train.columns, path, level + '_' + mode)