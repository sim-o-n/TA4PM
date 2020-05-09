"""
This file contains functions to compute temporal features on the activity
level for an event log.
"""

import datetime
from multiprocessing import Queue, Process
import pandas as pd
import copy
from features import do_kmeans
from util import print_time, get_steps, get_steps_seq


def time_feature(data: pd.DataFrame):
	"""
	Compute basic features for given dataset.

	:param data: event log
	:return: DataFrame
	"""
	print_time('activity time features')
	current_case = None

	activity_blueprint = {'caseID': 0, 'name': None, 'start': 0, 'end': 0,
						  'duration': 0, 'weekday': 0, 'start_am': 0,
						  'abs_lag': 0, 'parallel_activities': 0}

	case_collector = []
	acty_collector = {}
	case_base_time = 0

	# iterate through event log
	for idx, row in data.iterrows():
		current_act = row['activity_instance']

		if current_case != row['caseID']:

			# finish case and start over
			if current_case is not None:
				for a, v in acty_collector.items():
					acty_collector[a]['duration'] = (acty_collector[a]['end']
													 - acty_collector[a][
														 'start']).total_seconds()
					case_collector.append(v)
				acty_collector = {}
			current_case = row['caseID']
			case_base_time = row['timestamp']

		# add new activity instance
		if current_act not in acty_collector.keys():
			t0 = row['timestamp']
			acty_collector[current_act] = copy.deepcopy(activity_blueprint)
			acty_collector[current_act]['caseID'] = current_case
			acty_collector[current_act]['name'] = row['name'].strip()
			acty_collector[current_act]['start'] = t0
			acty_collector[current_act]['end'] = t0
			acty_collector[current_act]['abs_lag'] = (t0 -
													  case_base_time).total_seconds()
			if t0.hour < 12:
				# else by default
				acty_collector[current_act]['start_am'] = 1
			if t0.isoweekday() < 6:
				# else by default
				acty_collector[current_act]['weekday'] = 1
		else:
			acty_collector[current_act]['end'] = row['timestamp']

		if idx == len(data) - 1:
			for a, v in acty_collector.items():
				acty_collector[a]['duration'] = (acty_collector[a]['end']
												 - acty_collector[a][
													 'start']).total_seconds()
				case_collector.append(v)

	return pd.DataFrame(case_collector)


def multiprocess_parallel_activities(data: pd.DataFrame, job_num: int):
	"""
	Compute parallel executed activity instances in given dataset. The
	computation will be	split into the given number of jobs.

	:param data: activity log
	:param job_num: number of jobs
	"""
	print_time('parallel activities')
	activity_name = data['name'].unique()

	# compute for all activities
	for a in activity_name:
		print_time('calculate parallel activities for activity %s' % a)

		# get subset
		sub_data = data.loc[data['name'] == a]

		steps = get_steps(sub_data, job_num)
		print(steps)

		jobs = []
		out_q = Queue()

		# set time interval (=theta)
		delta = datetime.timedelta(days=1)

		# start all jobs
		for idx, r in enumerate(steps):
			p = Process(target=parallel_activities, args=(
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

		# update DataFrame
		for k, v in res.items():
			data.at[k, 'parallel_activities'] = v


def parallel_activities(data: pd.DataFrame, delta: datetime.timedelta, instance,
						range: list, out_q: Queue):
	"""
	This function is called by multiprocess_parallel_activities(). It computes
	the number of parallel executed activity instances in a given subset.

	:param data: dataset
	:param delta: time interval
	:param instance: current instance
	:param range: subset
	:param out_q: output queue
	"""

	# total number of activity instances
	size = len(data)

	# collect results
	r = {}

	for idx, row in data[range[0]:range[1]].iterrows():
		exclude_cases = data.loc[
			(data['start'] > row['end'] + delta) |
			(data['end'] < row['start'] - delta)
			]
		counter = size - len(exclude_cases)
		r[idx] = counter

	print_time('instance: %s' % instance, False)
	out_q.put(r)


def multiprocess_aggregation(data: pd.DataFrame, job_num: int):
	"""
	Aggregate enriched activity log to case log. The computation will be
	split into the given number of jobs.

	:param data: enriched activity log
	:param job_num: number of jobs
	:return: case log
	"""
	print_time('aggregate activities')

	case_ids = data['caseID'].unique()
	steps = get_steps_seq(case_ids, data, job_num)

	# prepare names
	activity_names = data['name'].unique()
	activity_names = ['_'.join(['act', str(i)]) for i in activity_names]

	feature = ['abs_lag', 'duration', 'start_am', 'weekday',
			   'parallel_activities']

	# generate blueprint
	case_tmp = {}
	for a in activity_names:
		case_tmp[a] = {'counter': 0}
		for f in feature:
			case_tmp[a][f] = 0

	jobs = []
	out_q = Queue()

	# start all jobs
	for idx, r in enumerate(steps):
		p = Process(target=aggregate,
					args=(data, case_tmp, r, out_q, idx))
		jobs.append(p)
		p.start()

	# merge jobs and collect results
	res = {}
	for i in range(len(steps)):
		res.update(out_q.get())

	for job in jobs:
		job.join()

	print_time('aggregate activities', start=False)
	print_time('merge results')

	return pd.DataFrame(list(res.values()))


def aggregate(data: pd.DataFrame, case_temp: dict, range: list, out_q: Queue,
			  instance: int):
	"""
	This function is called by multiprocess_aggregation(). It aggregates a
	given subset of activity instances.

	:param data: enirichted activity log
	:param case_temp: blueprint
	:param range: subset
	:param out_q: outout Queue
	:param instance: current instance
	"""

	def aggr(container: dict, case: int):
		"""
		Aggregate features of one case.

		:param container: collected features
		:param case: current case ID
		:return: aggregated case
		"""
		case = {'case': case}
		for act, features in container.items():
			for f, value in features.items():
				if f != 'counter':
					name = '_'.join(['avg', f, act])
					if features['counter'] > 0:
						case[name] = value / features['counter']
					else:
						case[name] = 0
		return case

	print('Instance %s and got %s cases.' % (instance, range))

	# store results
	r = {}
	current_case = None
	single_case_template = copy.deepcopy(case_temp)

	for idx, row in data[range[0]:range[1]].iterrows():

		current_activity = 'act_' + str(row['name'])

		if current_case is None or current_case != row['caseID']:
			# next case
			if current_case is None:
				pass
			else:
				case_aggregat = aggr(single_case_template, current_case)
				r[current_case] = case_aggregat
				single_case_template = copy.deepcopy(case_temp)
			current_case = row['caseID']


		single_case_template[current_activity]['counter'] += 1
		single_case_template[current_activity]['abs_lag'] += row['abs_lag']
		single_case_template[current_activity]['duration'] += row['duration']
		single_case_template[current_activity]['weekday'] += row['weekday']
		single_case_template[current_activity]['start_am'] += row['start_am']
		single_case_template[current_activity]['parallel_activities'] += row[
			'parallel_activities']

		if idx == range[1] - 1:
			# end last case in subset
			case_aggregat = aggr(single_case_template, current_case)
			r[current_case] = case_aggregat

	# add results to output
	out_q.put(r)
	print_time('Instance %s' % instance, start=False)


def find_cluster(data: pd.DataFrame, base_path: str, plot: bool):
	"""
	This function calls the k-means clustering for all features.

	:param data: case log
	:param base_path: path to save plots and cluster information
	:param plot: whether to generate plots or not
	"""
	print_time('activities kmeans')
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
