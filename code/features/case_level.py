"""
This file contains functions to compute temporal features on the case level
of an event log.
"""

import copy
from multiprocessing import Queue, Process
import pandas as pd
from features import do_LOF, do_kmeans
from util import print_time, progress, get_steps


def time_feature(data: pd.DataFrame, intervals: list) -> pd.DataFrame:
	"""
	Compute basic features for given dataset.

	:param data: event log
	:param intervals: progress intervals
	:return: case log
	"""
	print_time('case time features')

	all_cases = []
	current_case = None

	# features
	x = {'case': None, 'case_duration': 0, 'case_start_weekday': False,
		 'start': None, 'end': None, 'overlapping_cases': None,
		 'case_start_am':
			 True}

	# iterate over all events
	for idx, row in data.iterrows():

		if current_case is None or current_case != row['caseID']:

			# update case
			if current_case is not None:
				if data.iloc[idx - 1]['caseID'] == current_case:
					t = data.iloc[idx - 1]['timestamp']
					duration = (t - base_time).total_seconds()
					x2['end'] = t
				else:
					duration = 0
					x2['end'] = base_time

				x2['case_duration'] = duration
				# close case
				all_cases.append(x2)
			# new case

			x2 = copy.deepcopy(x)
			base_time = row['timestamp']
			current_case = row['caseID']
			x2['case'] = current_case
			x2['case_start_weekday'] = base_time.isoweekday() < 6
			x2['case_start_am'] = base_time.hour < 12
			x2['start'] = base_time

		elif idx == len(data) - 1:
			# close last case in dataset
			if data.iloc[idx - 1]['caseID'] == current_case:
				t = data.iloc[idx - 1]['timestamp']
				duration = (t - base_time).total_seconds()
				x2['end'] = t
			else:
				duration = 0
				x2['end'] = base_time
			x2['case_duration'] = duration
			all_cases.append(x2)

		progress(len(data), idx + 1, 'case feature', intervals)

	return pd.DataFrame(all_cases)


def multiprocess_overlapping_cases(data: pd.DataFrame, job_num: int):
	"""
	Compute parallel executed case instances in given dataset. The
	computation will be	split into the given number of jobs.

	:param data: case log
	:param job_num: number of jobs
	"""
	print_time('parallel cases %s' % len(data))

	# get split points
	steps = get_steps(data, job_num)

	print(steps)
	jobs = []
	out_q = Queue()

	# start all jobs
	for idx, r in enumerate(steps):
		p = Process(target=overlapping_cases,
					args=(data, idx + 1, r, out_q))
		jobs.append(p)
		p.start()

	# collect results
	res = {}
	for i in range(len(steps)):
		res.update(out_q.get())

	# collect processes
	for job in jobs:
		job.join()

	# update case log
	for k, v in res.items():
		data.at[k, 'overlapping_cases'] = v


def overlapping_cases(data: pd.DataFrame, instance: int, range: list,
					  out_q: Queue):
	"""
	This function is called by multiprocess_overlapping_cases(). It computes
	the number of parallel executed case instances in a given subset.

	:param data: case log
	:param instance: current instace
	:param range: subset
	:param out_q: output queue
	"""

	# total number of cases
	size = len(data)

	# collect results
	r = {}

	for idx, row in data[range[0]:range[1]].iterrows():

		exclude_cases = data.loc[
			(data['start'] > row['end']) |
			(data['end'] < row['start'])]

		counter = size - len(exclude_cases)
		r[idx] = counter

		# log progress
		progress(len(data[range[0]:range[1]]), len(r),
				 'overlapping cases - instance: %s' % instance)

	# collect final results
	out_q.put(r)


def mark_outlier(data: pd.DataFrame, base_path: str):
	"""
	This function calls the LOF computation for each feature.

	:param data: case log
	:param base_path: path to location to save plots
	"""
	print_time('case mark outlier')
	columns = []

	for c in list(data.columns):
		# ignore selected columns
		if 'case_duration' in c or 'overlapping_cases' in c:
			columns.append(c)
		else:
			pass

	# call LOF computation
	for c in columns:
		do_LOF(data, c, base_path, plot=True)


def find_cluster(data: pd.DataFrame, base_path: str, plot: bool):
	"""
	This function calls the k-means clustering for all features.

	:param data: case log
	:param base_path: path to save plots and cluster information
	:param plot: whether to generate plots or not
	"""
	print_time('case kmeans')
	columns = []

	for c in list(data.columns):
		# ignore selected columns
		if 'case_duration' in c or 'overlapping_cases' in c:
			columns.append(c)
		else:
			pass

	# call k-means computation
	for c in columns:
		do_kmeans(data, c, base_path, plot)
