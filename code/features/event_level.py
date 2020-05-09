"""
This file contains functions to compute temporal features on the event
level for an event log.
"""
import copy
import pandas as pd
import datetime

from features import do_LOF, do_kmeans
from util import print_time, get_steps, get_steps_seq
from multiprocessing import Process, Queue


def multiprocess_time_feature(log_data: pd.DataFrame, job_num: int):
	"""
	Compute basic features for given dataset. The computation will be split
	into the given number of jobs.

	:param log_data: event log
	:param job_num: number of jobs
	"""
	print_time('time features')
	case_ids = log_data['caseID'].unique()

	# convert variable type of timestamp
	log_data['timestamp'] = pd.to_datetime(log_data.loc[:, 'timestamp'],
										   utc=True, format='%Y-%m-%d '
															'%H:%M:%S')

	# get split points
	steps = get_steps_seq(case_ids, log_data, job_num)

	jobs = []
	out_q = Queue()

	# start jobs
	for idx, r in enumerate(steps):
		p = Process(target=multi_time_feature,
					args=(log_data, r, out_q, idx))
		jobs.append(p)
		p.start()

	# collect results
	res = {}
	for i in range(len(steps)):
		res.update(out_q.get())

	for job in jobs:
		job.join()

	print_time('time features', start=False)
	print_time('merge results')

	# add features to DataFrame
	for idx, data in res.items():
		for attr, v in data.items():
			log_data.at[idx, attr] = v


def multi_time_feature(data: pd.DataFrame, range: tuple, out_q: Queue,
					   instance: int):
	"""
	This function is called by multiprocess_time_feature(). It calculates the
	features for the given subset.

	:param data: event data
	:param range: subset
	:param out_q: output queue
	:param instance: job number
	"""
	print('Instance %s and got %s cases.' % (instance, range))

	# collect results for single events
	r = {}

	current_case = None
	# iterate through all cases aka rows in the DataFrame
	for idx, row in data[range[0]:range[1]].iterrows():
		case_row = {}

		t0 = row['timestamp']

		if current_case is None or current_case != row['caseID']:

			# update new case
			base_time = row['timestamp']
			current_case = row['caseID']
			pre_lag = 0
		else:

			t1 = data.iloc[idx - 1]['timestamp']

			pre_lag = (t0 - t1).total_seconds()

		# post lag feature
		if idx + 1 < len(data):
			if data.iloc[idx + 1]['caseID'] != current_case:
				post_lag = 0
			else:
				t1 = data.iloc[idx + 1]['timestamp']

				post_lag = (t1 - t0).total_seconds()
		else:
			post_lag = 0

		case_row['pre_lag'] = pre_lag
		case_row['post_lag'] = post_lag
		case_row['abs_lag'] = (t0 - base_time).total_seconds()

		if t0.isoweekday() < 6:
			case_row['weekday'] = 1
			case_row['weekend'] = 0
		else:
			case_row['weekday'] = 0
			case_row['weekend'] = 1

		# collect features for each event
		r[idx] = case_row

	# add result to output queue
	out_q.put(r)

	msg = 'Instance %s' % instance
	print_time(msg, start=False)


def multiprocess_overlapping_events(log_data: pd.DataFrame, job_num: int):
	"""
	Compute parallel executed events of given dataset. The computation will be
	split into the given number of jobs.

	:param log_data: event log
	:param job_num: number of jobs
	"""

	print_time('parallel events')

	# get all event names
	event_names = log_data['name'].unique()

	# convert variable type of timestamp
	log_data['timestamp'] = pd.to_datetime(log_data.loc[:, 'timestamp'],
										   format='%Y-%m-%d %H:%M:%S')

	# iterate through all event names
	for e in event_names:
		print_time('calculate parallel events for %s' % e)

		# get subset
		sub_data = log_data.loc[log_data['name'] == e]
		sub_data = sub_data.sort_values(by=['timestamp'])

		steps = get_steps(sub_data, job_num)
		print(steps)

		# collect jobs and results
		jobs = []
		out_q = Queue()

		# set time interval (= theta)
		delta = datetime.timedelta(days=1)

		# start different jobs
		for idx, r in enumerate(steps):
			p = Process(target=overlapping_events, args=(
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

		for k, v in res.items():
			log_data.at[k, 'parallel_events'] = v


def overlapping_events(data: pd.DataFrame, delta: datetime.timedelta,
					   instance: int, range: list, out_q: Queue):
	"""
	This function is called by multiprocess_overlapping_events(). It computes
	the number of parallel executed events for a given subset.

	:param data: dataset
	:param delta: time interval
	:param instance: job number
	:param range: subset of dataset
	:param out_q: output queue
	"""

	# collect results
	r = {}
	for idx, row in data[range[0]:range[1]].iterrows():

		l_bound = data['timestamp'].searchsorted(row['timestamp'] - delta)
		u_bound = data['timestamp'].searchsorted(row['timestamp'] + delta,
												 'right')
		counter = u_bound[0] - l_bound[0]

		r[idx] = counter

	out_q.put(r)


def multiprocess_aggregation(data: pd.DataFrame, job_num: int):
	"""
	Aggregate enriched event log to case log. The computation will be
	split into the given number of jobs.

	:param data: event log
	:param job_num: number of jobs
	:return: case log
	"""
	print_time('aggregate events')
	print('Lenght dataset %s' % len(data))

	case_ids = data['caseID'].unique()

	steps = get_steps_seq(case_ids, data, job_num)
	print(steps)

	events = data['name'].unique()

	# feature types
	feature = ['abs_lag', 'pre_lag', 'post_lag', 'weekday', 'weekend',
			   'parallel_events']

	# blueprint to collect single events
	case_tmp = {}
	for e in events:
		case_tmp[e] = {'counter': 0}
		for f in feature:
			case_tmp[e][f] = 0

	jobs = []
	out_q = Queue()

	# start all jobs
	for idx, r in enumerate(steps):
		p = Process(target=aggregate,
					args=(data, case_tmp, r, out_q, idx))
		jobs.append(p)
		p.start()

	# collect results
	res = {}
	for i in range(len(steps)):
		res.update(out_q.get())

	for job in jobs:
		job.join()

	print_time('time features', start=False)
	print_time('merge results')

	# return case log
	return pd.DataFrame(list(res.values()))


def aggregate(data: pd.DataFrame, case_temp: dict, range, out_q, instance):
	"""
	This function is called by multiprocess_aggregation(). It aggregates a
	given subset of events.

	:param data: enriched event log
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
		for event, features in container.items():
			for f, value in features.items():
				if f == 'counter':
					# indicate whether an event was part of a case or not
					#  -> baseline
					name = '_'.join(['involved', event])
					if value > 0:
						case[name] = True
					else:
						case[name] = False
				else:
					name = '_'.join(['avg', f, event])
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

	# iterate over data subset
	for idx, row in data[range[0]:range[1]].iterrows():

		current_event = row['name']

		if current_case is None or current_case != row['caseID']:
			# next case
			if current_case is None:
				pass
			else:
				# end current case
				case_aggregat = aggr(single_case_template, current_case)
				r[current_case] = case_aggregat
				# start new case
				single_case_template = copy.deepcopy(case_temp)

			current_case = row['caseID']

		single_case_template[current_event]['counter'] += 1
		single_case_template[current_event]['abs_lag'] += row['abs_lag']
		single_case_template[current_event]['pre_lag'] += row['pre_lag']
		single_case_template[current_event]['post_lag'] += row['post_lag']
		single_case_template[current_event]['weekday'] += row['weekday']
		single_case_template[current_event]['weekend'] += row['weekend']
		single_case_template[current_event]['parallel_events'] += row[
			'parallel_events']

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
	"""
	print_time('events: mark outlier')
	columns = []

	for c in list(data.columns):
		# ignore selected columns
		if 'case' in c or 'involved' in c:
			pass
		else:
			columns.append(c)

	# call LOF computation
	for c in columns:
		do_LOF(data, c, base_path, plot=False)


def find_cluster(data: pd.DataFrame, base_path: str, plot: bool):
	"""
	This function calls the k-means clustering for all features.

	:param data: case log
	:param base_path: path to save plots and cluster information
	:param plot: whether to generate plots or not
	"""
	print_time('events kmeans')
	columns = []

	for c in list(data.columns):
		# ignore selected columns
		if 'case' in c or 'involved' in c or 'weekday' in c or 'weekend' in c:
			pass
		else:
			columns.append(c)

	# call k-means computation
	for c in columns:
		do_kmeans(data, c, base_path, plot)
