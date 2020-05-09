"""
This file contains functions, to import XES event logs and to assign activity
instances.
"""

from datetime import datetime
import pandas as pd
import xml.etree.ElementTree as et
from util import progress, print_time


def read_xml(file: str, limit=0):
	"""
	This function read a XES event log and returns it as a pandas DataFrame.
	If no case limit is set, all cases will be imported.

	:param file: path to event log in XES format
	:param limit: number of cases
	:return: pandas DataFrame
	"""
	tree = et.parse(file)
	data = tree.getroot()

	l_trace = []
	l_name = []
	l_timestamp = []
	l_lifecycle = []
	l_resource = []
	l_activity_instance = []

	# find all traces and extract events
	traces = data.findall('trace')

	print('Found %s traces in the log' % len(traces))

	# iterate through traces
	for idx, trace in enumerate(traces):
		trace_id = None

		# trace attributes
		for a in trace.findall('string'):
			if a.attrib['key'] == 'concept:name':
				trace_id = a.attrib['value']

		# events
		for event in trace.iter('event'):
			# attributes
			res = False
			l_trace.append(trace_id)
			l_activity_instance.append(None)

			for a in event:
				if a.attrib['key'] == 'time:timestamp':
					ts = a.attrib['value']
					ts = ts[:19]
					format = '%Y-%m-%dT%H:%M:%S'
					ts = datetime.strptime(ts, format)
					l_timestamp.append(ts)
				elif a.attrib['key'] == 'concept:name':
					l_name.append(a.attrib['value'])
				elif a.attrib['key'] == 'lifecycle:transition':
					l_lifecycle.append(a.attrib['value'])
				elif a.attrib['key'] == 'org:resource':
					if 'n/a' not in a.attrib['value']:
						l_resource.append(a.attrib['value'])
						res = True

			if not res:
				l_resource.append(None)

		# stop import if case limit is reached
		if idx == limit - 1 and limit > 0:
			break

		# track process
		progress(len(traces), idx, 'import traces', ['0.10', '0.20', '0.30',
													 '0.40', '0.50', '0.60',
													 '0.70', '0.80', '0.90'])
	data = {'caseID': l_trace,
			'name': l_name,
			'transition': l_lifecycle,
			'timestamp': l_timestamp,
			'resource': l_resource,
			'activity_instance': l_activity_instance}

	# build DataFrame
	df = pd.DataFrame(data)
	print('Import .... DONE')
	return df


def set_activity_instances(df: pd.DataFrame):
	"""
	This function maps events and activity instances based on a first-come,
	first-serves approach.

	:param df: event log
	"""
	print_time('set activity instance')
	end_transitions = ['autoskip', 'manualskip', 'complete',
					   'withdraw', 'ate_abort', 'pi_abort']

	trace = df.loc[0, 'caseID']
	activity = 0
	instances = {}
	for idx, row in df.iterrows():
		if row['caseID'] != trace:
			trace = row['caseID']
			instances = {}

		if row['name'] not in instances.keys():
			activity += 1
			instances[row['name']] = activity

		df.at[idx, 'activity_instance'] = instances[row['name']]

		if row['transition'].lower() in end_transitions:
			instances.pop(row['name'], None)

	print_time('set activity instance', False)


def get_concept_names(df: pd.DataFrame):
	return df['concept:name'].unique()
