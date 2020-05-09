# Temporal Analysis for Process Mining

This codebase can be split into two parts: feature computation and decision tree classification for temporal root cause analysis in conformance checking.

## General settings:
- the code was developed for Python 3.5

## Feature Computation:
- parameters and other settings (such as working-directory) must be set in the `feature_config.ini` in advanced:
    - base_path: working-directory
    - log_file: path to XES event log
    - report_file: path to conformance report (CSV)
    - event_sets: path to event sets (CSV)
    - case_limit: number of cases to import (set -1 to import all)
    - number_jobs: number of parallel jobs
    - plots: True or False
    - feature_level: process granularity(ies) e.g. event,set,case (NO spaces!)
- the conformance report must only contain the case information. If necessary, leading and subsequent rows have to be deleted in advance.
- event sets have to be stored in a `.csv` file. Events of one set have to be separated by `|`
- use the `compute_features.py` script to start the feature computation


## Classification:
- use the `classify.py` script to start the classification task
