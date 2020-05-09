"""
This file contains functions to compute temporal features on the event
set level for an event log.
"""

import copy
import datetime
from multiprocessing import Queue, Process

import pandas as pd

from features import do_LOF, do_kmeans
from util import print_time, get_steps, get_steps_seq


def time_feature(data: pd.DataFrame, sets: list) -> pd.DataFrame:
	"""
	Compute basic features for given dataset.

	:param data: event log
	:param sets: list of event sets
	:return: DataFrame
	"""

	def get_set(event: str, sets: list):
		"""
		Get number of set, the given event belongs to.

		:param event: event name
		:param sets: list of event sets
		:return: set number
		"""
		for idx, s in enumerate(sets):
			if event.strip() in s:
				return idx
		# alert, if the given event is not part of any set
		print('####### %s #######' % event)

	current_case = None
	current_set = None
	set_coll = {}

	all_sets = []

	# iterate through all events in the log
	for idx, row in data.iterrows():
		event = row['name']
		row_set = get_set(event, sets)
		if current_case != row['caseID'] or current_set != row_set:

			if current_case != row['caseID']:
				# case changed
				if current_case is None:
					pass
				else:
					# close current set
					if get_set(data.iloc[idx - 1]['name'], sets) == \
							current_set:
						set_duration = (data.iloc[idx - 1]['timestamp']
										- set_t0).total_seconds()

					else:
						set_duration = 0

					set_coll['duration'] = set_duration
					set_coll['end'] = data.iloc[idx - 1]['timestamp']
					# collect set
					all_sets.append(set_coll)

				case_base_time = row['timestamp']
				current_case = row['caseID']


			else:
				# set changed
				if current_set is None:
					pass
				else:
					# close set
					set_duration = row['timestamp'] - set_t0
					set_coll['end'] = data.iloc[idx - 1]['timestamp']
					set_coll['duration'] = set_duration.total_seconds()
					# collect set
					all_sets.append(set_coll)

			current_set = row_set
			set_coll = {}
			set_t0 = row['timestamp']
			set_coll['start'] = set_t0
			set_coll['caseID'] = current_case
			set_coll['set_name'] = row_set
			if set_t0.isoweekday() < 6:
				set_coll['weekday'] = 1
			else:
				set_coll['weekday'] = 0
			if set_t0.hour < 12:
				set_coll['start_am'] = 1
			else:
				set_coll['start_am'] = 0
			set_coll['abs_lag'] = (set_t0 -
								   case_base_time).total_seconds()

	return pd.DataFrame(all_sets)


def multiprocess_parallel_event_sets(data: pd.DataFrame, job_num: int):
	"""
	Compute parallel executed event sets of given dataset. The computation
	will be	split into the given number of jobs.

	:param data: event log
	:param job_num: number of jobs
	"""

	print_time('parallel event sets')

	data['parallel_sets'] = [0] * len(data)

	set_names = data['set_name'].unique()
	print('Set names: %s' % set_names)

	# iterate through sets
	for s in set_names:
		print_time('calculate parallel events sets for set_%s' % s)
		sub_data = data.loc[data['set_name'] == s]

		# get split points
		steps = get_steps(sub_data, job_num)
		print(steps)

		# collect jobs and results
		jobs = []
		out_q = Queue()

		# set time interval (= theta)
		delta = datetime.timedelta(days=1)

		# start jobs
		for idx, r in enumerate(steps):
			p = Process(target=parallel_event_sets, args=(
				sub_data, delta,
				idx + 1,
				r, out_q))
			jobs.append(p)
			p.start()

		# collect results
		res = {}
		for i in range(len(steps)):
			res.update(out_q.get())

		# collect processes
		for job in jobs:
			job.join()

		# update data
		for k, v in res.items():
			data.at[k, 'parallel_sets'] = v


def parallel_event_sets(data: pd.DataFrame, delta: datetime.timedelta,
						instance: int, range: list, out_q: Queue):
	"""
	This function is called by multiprocess_parallel_event_sets(). It computes
	the number of parallel executed event sets for a given subset.

	:param data: dataset
	:param delta: time interval
	:param instance: current instance
	:param range: subset
	:param out_q: output queue
	"""

	# total number of event sets
	size = len(data)

	# collect results
	r = {}

	for idx, row in data[range[0]:range[1]].iterrows():
		exclude_cases = data.loc[
			(data['start'] > row['end'] + delta) |
			(data['end'] < row['start'] - delta)]
		counter = size - len(exclude_cases)

		r[idx] = counter

	print_time('instance: %s' % instance, False)

	# collect results
	out_q.put(r)


def multiprocess_aggregation(data: pd.DataFrame, job_num: int):
	"""
	Aggregate enriched event set log to case log. The computation will be
	split into the given number of jobs.

	:param data: enriched event set log
	:param job_num: number of jobs
	:return: case log
	"""
	print_time('aggregate event sets')

	case_ids = data['caseID'].unique()
	steps = get_steps_seq(case_ids, data, job_num)

	# get set names
	sets = data['set_name'].unique()
	set_names = ['_'.join(['set', str(i)]) for i in sets]

	# feature types
	feature = ['abs_lag', 'duration', 'start_am', 'weekday', 'parallel_sets']

	# blueprint to collect event sets in one case
	case_tmp = {}
	for s in set_names:
		case_tmp[s] = {'counter': 0}
		for f in feature:
			case_tmp[s][f] = 0

	jobs = []
	out_q = Queue()

	# start all jobs
	for idx, r in enumerate(steps):
		p = Process(target=aggregate,
					args=(data, case_tmp, r, out_q, idx))
		jobs.append(p)
		p.start()

	# collect results and jobs
	res = {}
	for i in range(len(steps)):
		res.update(out_q.get())

	for job in jobs:
		job.join()

	print_time('time features', start=False)
	print_time('merge results')

	return pd.DataFrame(list(res.values()))

def aggregate(data: pd.DataFrame, case_temp: dict, range: list, out_q: Queue,
			  instance: int):
	"""
	This function is called by multiprocess_aggregation(). It aggregates a
	given subset of event sets.

	:param data: enriched event set log
	:param case_temp: blueprint
	:param range: subset
	:param out_q: output Queue
	:param instance: job number
	"""

	def aggr(container: dict, case: int):
		"""
		Aggregate features of one case.

		:param container: collected features
		:param case: current case
		:return: aggregated case
		"""
		case = {'case': case}
		for set, features in container.items():
			for f, value in features.items():
				if f != 'counter':
					name = '_'.join(['avg', f, set])
					if features['counter'] > 0:
						case[name] = value / features['counter']
					else:
						case[name] = 0
		return case

	print('Instance %s and got %s cases.' % (instance, range))

	# collect results
	r = {}

	current_case = None
	single_case_template = copy.deepcopy(case_temp)

	for idx, row in data[range[0]:range[1]].iterrows():

		current_set = 'set_' + str(row['set_name'])

		if current_case is None or current_case != row['caseID']:
			# next case
			if current_case is None:
				pass
			else:
				case_aggregat = aggr(single_case_template, current_case)
				r[current_case] = case_aggregat
				# new case
				single_case_template = copy.deepcopy(case_temp)
			current_case = row['caseID']


		single_case_template[current_set]['counter'] += 1
		single_case_template[current_set]['abs_lag'] += row['abs_lag']
		single_case_template[current_set]['duration'] += row['duration']
		single_case_template[current_set]['weekday'] += row['weekday']
		single_case_template[current_set]['start_am'] += row['start_am']
		single_case_template[current_set]['parallel_sets'] += row[
			'parallel_sets']

		if idx == range[1] - 1:
			# end last case in subset
			case_aggregat = aggr(single_case_template, current_case)
			r[current_case] = case_aggregat

	# add to output queue
	out_q.put(r)
	msg = 'Instance %s' % instance
	print_time(msg, start=False)


def mark_outlier(data: pd.DataFrame, base_path: str):
	"""
	This function calls the LOF computation for each feature.

	:param data: case log
	:param base_path: path to save plots
	:return:
	"""
	print_time('event set mark outlier')
	columns = []

	for c in list(data.columns):
		# ignore selected columns
		if 'case' in c or 'involved' in c:
			pass
		else:
			columns.append(c)

	# call LOF computations
	for c in columns:
		do_LOF(data, c, base_path, plot=False)


def find_cluster(data: pd.DataFrame, base_path: str, plot: bool):
	"""
	This function calls the k-means clustering for all features.

	:param data: case log
	:param base_path: path to save plots and cluster information
	:param plot: whether to generate plots or not
	"""
	print_time('event sets kmeans')
	columns = []

	for c in list(data.columns):
		# ignore selected columns
		if 'case' in c or 'involved' in c or 'weekday' in c or 'weekend' in c \
				or 'start_am' in c:
			pass
		else:
			columns.append(c)

	# call k-means computation
	for c in columns:
		do_kmeans(data, c, base_path, plot)
