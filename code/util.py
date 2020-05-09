"""
This file contains functions that are used all across this project. These
functions are used to:
- log computational progress
- print timestamps
- split data structures
"""
from datetime import datetime
import pandas as pd


def progress(total: int, current: int, msg: str,
			 intervals=['0.25', '0.50', '0.75']):
	"""
	This function can be used to monitor the progress of any computation. It
	can print messages for user defined intervals.

	:param total: total number of steps
	:param current: current step
	:param msg: message to print
	:param intervals: progress intervals
	"""
	progress = current / total
	pre_progress = (current - 1) / total

	# format progress
	progress = str('%.2f' % progress)
	pre_progress = str('%.2f' % pre_progress)
	if progress != '1.00' and progress != pre_progress:
		if progress in intervals:
			print('%s%% ... %s DONE' % ((float(progress) * 100), msg))
	else:
		if progress != pre_progress:
			print('100%% ... %s DONE ' % msg)


def print_time(msg: str = None, start: bool = True):
	"""
	This function prints the current system time and if provided a message.

	:param msg: message to print
	:param start: whether the timestamp indicates the beginning of something
	or not
	"""
	if start:
		state = 'START'
	else:
		state = 'END'

	print('%s %s at %s' % (state, msg, datetime.now()))


def get_steps(data: pd.DataFrame, jobs: int):
	"""
	This function returns pairs of lower and upper bounds to split a given
	DataFrame.

	:param data: data to split
	:param jobs: number of jobs
	:return: list of pairs
	"""

	step_size = int(len(data) // jobs)
	if step_size == 0:
		step_size = 1
	steps = range(0, len(data), step_size)
	pairs = []
	for s in steps[0:jobs]:
		pairs.append((s, s + step_size))

	pairs[-1] = (pairs[-1][0], len(data) + 1)
	return pairs


def get_steps_seq(elem: list, data: pd.DataFrame, jobs: int):
	"""
	This function returns pairs of indices of a elements in a DataFrame to
	split.

	:param elem: data sequence
	:param data: DataFrame
	:param jobs: number of splits
	:return: list of pairs of indices
	"""

	step_size = int(len(elem) // (jobs - 1))
	if step_size == 0:
		step_size = 1

	splits = []
	for i in elem[::step_size]:
		splits.append(i)

	idxs = []
	for s1, s2 in zip(splits, splits[1:]):
		idxs.append((min(data.index[data['caseID'] == s1].tolist()),
					 min(data.index[data['caseID'] == s2].tolist())))

	# ensure that all elements are included
	idxs.append((min(data.index[data['caseID'] == splits[-1]].tolist()),
				 len(data)))

	return idxs
