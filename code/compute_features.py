"""
This is the main script for the feature computation. Use the file
feature_config.ini to set certain parameters.
"""

from functools import reduce
from configparser import ConfigParser
import pandas as pd
from features import case_level, event_set_level, event_level, \
	import_report, \
	activity_level
from log import read_xml, set_activity_instances
from util import print_time

# %% settings
"""
Settings:
Parameters, set in the corresponding configuration fill will be read.
"""
pd.set_option('display.expand_frame_repr', False)

# set intervals to log computation progress
intervals = ['0.10', '0.20', '0.30', '0.40', '0.50',
			 '0.60', '0.70', '0.80', '0.90']

# set and read config file
config = ConfigParser()
config.read('feature_config.ini')

# read configurations
base_path = config.get('main', 'base_path')
file_name = config.get('main', 'log_file')
report_file = config.get('main', 'report_file')
event_sets = config.get('main', 'event_sets')

# case limit and number of parallel jobs
case_limit = int(config.get('main', 'case_limit'))
jobs = int(config.get('main', 'number_jobs'))

# whether results of k-means clustering should be plotted or not
do_plots = config.getboolean('main', 'plots')

# set process granularities to examine
feature_classes = config.get('features', 'feature_level').split(',')

# %% read data
"""
Import event log:
In this section the given event log will be imported and activity instances 
will be set.
"""
print_time('feature computation')

# import event log
if case_limit < 0:
	log_data = read_xml(base_path + file_name)
else:
	log_data = read_xml(base_path + file_name, limit=case_limit)

set_activity_instances(log_data)

"""
Set response variable
"""
# import conformance report
conformance = import_report(base_path + report_file)
print('Length conformance %s' % len(conformance))

# %% feature computation
"""
Compute features:
In the following, features for all chosen process granularities will be 
computed.
"""
# collect aggregated data
df_list = []

if 'event' in feature_classes:
	log_data['parallel_events'] = [0] * len(log_data)
	log_data['post_lag'] = [0] * len(log_data)
	log_data['pre_lag'] = [0] * len(log_data)
	log_data['abs_lag'] = [0] * len(log_data)
	log_data['weekday'] = [0] * len(log_data)
	log_data['weekend'] = [0] * len(log_data)

	# compute basic feature
	event_level.multiprocess_time_feature(log_data, jobs)

	# compute parallel events
	event_level.multiprocess_overlapping_events(log_data, jobs)

	# aggregate to case log
	agr_event_data = event_level.multiprocess_aggregation(log_data, jobs)

	# cluster features
	event_level.find_cluster(agr_event_data, base_path, do_plots)

	# add feature level to merge list
	df_list.append(agr_event_data)

	# export single feature level to csv
	agr_event_data.to_csv(base_path + 'features_events_agr.csv', index=False)

if 'set' in feature_classes:
	# import event sets
	sets_df = pd.read_csv(base_path + event_sets)
	sets = []
	for idx, row in sets_df.iterrows():
		sets.append(row['set'].split('|'))

	print('Length data %s' % len(log_data))
	print('Set: %s' % len(sets))

	event_set_data = event_set_level.time_feature(log_data, sets)

	event_set_level.multiprocess_parallel_event_sets(
		event_set_data, jobs)

	agr_event_set_data = event_set_level.multiprocess_aggregation(
		event_set_data, jobs)

	event_set_level.find_cluster(agr_event_set_data, base_path, do_plots)

	df_list.append(agr_event_set_data)
	agr_event_set_data.to_csv(base_path + 'features_event_sets_agr.csv',
							  index=False)

if 'activity' in feature_classes:

	# compute basic features
	activity_data = activity_level.time_feature(log_data)

	# compute parallel executed activity instances
	activity_level.multiprocess_parallel_activities(activity_data, jobs)

	# aggregate to case log
	agr_activity_data = activity_level.multiprocess_aggregation(activity_data,
																jobs)

	# compute clusters
	activity_level.find_cluster(agr_activity_data, base_path, do_plots)

	# add to merge list
	df_list.append(agr_activity_data)

	# export single feature level
	agr_activity_data.to_csv(base_path + 'features_activity_agr.csv',
							 index=False)

if 'case' in feature_classes:

	# compute basic features
	case_data = case_level.time_feature(log_data, intervals)

	# compute parallel case instances
	case_level.multiprocess_overlapping_cases(case_data, jobs)

	# compute clusters
	case_level.find_cluster(case_data, base_path, do_plots)

	# add to merge list
	df_list.append(case_data)

	# save single feature level
	case_data.to_csv(base_path + 'features_cases_agr.csv',
					 index=False)

# %% merge data sets
"""
Merge data sets:

In this step, all computed feature level will be merged into a single case log.
In addition, the response variable will be added to the dataset.
"""
# add response variable
df_list.append(conformance)

# check if all sub-case-logs contain a case column
for i, c in enumerate(df_list):
	if 'case' not in list(c.columns):
		print('not case %s' % i)

print_time('merge data sets')

# merge case logs on the case attribute
df_merged = reduce(lambda left, right: pd.merge(left, right, on=['case']),
				   df_list)

# %%  export
"""
Save features to csv file.
"""

# export
df_merged.to_csv(base_path + 'features.csv', index=False)
print_time('feature computation', False)
