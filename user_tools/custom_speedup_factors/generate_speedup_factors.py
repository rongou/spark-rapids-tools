#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spark RAPIDS speedup factor generation script"""

import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

parser = argparse.ArgumentParser(description="Speedup Factor Analysis")
parser.add_argument("--cpu", type=lambda s: [str(item) for item in s.split(',')],
                    help="List of directories of CPU profiler logs separated by commas", required=True)
parser.add_argument("--gpu", type=lambda s: [str(item) for item in s.split(',')],
                    help="List of directories of GPU profiler logs separated by commas", required=True)
parser.add_argument("--output", type=str, help="Filename for custom speedup factors", required=True)
parser.add_argument("--verbose", action="store_true", help="flag to generate full verbose output for logging raw node results")
parser.add_argument("--chdir", action="store_true", help="flag to change to work dir that's the script located")
args = parser.parse_args()

cpu_dirs = args.cpu
gpu_dirs = args.gpu
output = args.output
verbose = args.verbose

cpu_stage_log = {}
gpu_stage_log = {}
cpu_duration = 0.0
gpu_duration = 0.0

min_speedup = 1.0

if args.chdir:
    # Change to work dir that's the script located
    os.chdir(os.path.dirname(__file__))

# CPU log parsing
for dir_index, cpu_dir in enumerate(cpu_dirs):
    for app in os.listdir(cpu_dir):

        # - figure out query from application_info.csv
        app_info = pd.read_csv(cpu_dir + "/" + app + "/application_information.csv")
        app_name = str(dir_index) + "." + app_info.loc[0]["appName"]
        cpu_duration = cpu_duration + app_info.loc[0]["duration"]
        cpu_stage_log[app_name] = {}

        # - load wholestagecodegen_mapping.csv into a dictionary for lookups (CPU only)
        mapping_info = pd.read_csv(cpu_dir + "/" + app + "/wholestagecodegen_mapping.csv")
        mapping_info = mapping_info.groupby(['SQL Node'])['Child Node'].apply(','.join).reset_index()

        # - process sql_plan_metrics_for_application.csv
        #   - load in "duration" (CPU)
        #   - replace WholeStageCodegen (CPU only) with list of operators from mapping lookup file
        #     - mapping_info.parent = sql_times.nodeName
        cpu_sql_info = pd.read_csv(cpu_dir + "/" + app + "/sql_plan_metrics_for_application.csv")
        cpu_sql_times = cpu_sql_info[cpu_sql_info["name"] == "duration"]
        cpu_sql_combined = cpu_sql_times.set_index('nodeName').join(mapping_info.set_index('SQL Node'), how='left')

        #  - parse WholeStageCodegen durations with child node mapping
        cpu_sql_times_df = cpu_sql_combined[['Child Node', 'total']]

        for index, row in cpu_sql_times_df.iterrows():
            operators = str(row['Child Node']).split(',')
            duration = row['total']/len(operators)/1000.0
            for operator in operators:
                if operator in cpu_stage_log[app_name]:
                    cpu_stage_log[app_name][operator] = cpu_stage_log[app_name][operator] + duration
                else:
                    cpu_stage_log[app_name][operator] = duration

        # - parse top-level execs from sql_to_stage_information.csv
        cpu_stage_info = pd.read_csv(cpu_dir + "/" + app + "/sql_to_stage_information.csv")
        cpu_stage_times = cpu_stage_info[['Stage Duration', 'SQL Nodes(IDs)']]
        cpu_stage_times_df = cpu_stage_times.dropna()

        for index, row in cpu_stage_times_df.iterrows():
            node_list = str(row['SQL Nodes(IDs)'])
            operators = node_list.split(',')
            duration = row['Stage Duration']/(len(operators)-node_list.count("WholeStageCodegen"))

            for operator in operators:
                if "WholeStageCodegen" in operator:
                    continue

                op_key = operator.split('(')[0]
                if op_key in cpu_stage_log[app_name]:
                    cpu_stage_log[app_name][op_key] = cpu_stage_log[app_name][op_key] + duration
                else:
                    cpu_stage_log[app_name][op_key] = duration

# GPU log parsing
for dir_index, gpu_dir in enumerate(gpu_dirs):
    for app in os.listdir(gpu_dir):

        # - figure out query from application_info.csv
        app_info = pd.read_csv(gpu_dir + "/" + app + "/application_information.csv")
        app_name = str(dir_index) + "." + app_info.loc[0]["appName"]
        gpu_duration = gpu_duration + app_info.loc[0]["duration"]
        gpu_stage_log[app_name] = {}

        # - process sql_to_stage_information.csv to get stage durations
        # - split up duration by operators listed in each stage
        gpu_stage_info = pd.read_csv(gpu_dir + "/" + app + "/sql_to_stage_information.csv")
        gpu_stage_times = gpu_stage_info[['Stage Duration', 'SQL Nodes(IDs)']]

        for index, row in gpu_stage_times.iterrows():
            operators = str(row['SQL Nodes(IDs)']).split(',')
            duration = row['Stage Duration']/len(operators)
            for operator in operators:
                op_key = operator.split('(')[0]
                if op_key in gpu_stage_log[app_name]:
                    gpu_stage_log[app_name][op_key] = gpu_stage_log[app_name][op_key] + duration
                else:
                    gpu_stage_log[app_name][op_key] = duration

cpu_stage_durations = {}
gpu_stage_durations = {}
cpu_stage_total = 0.0
gpu_stage_total = 0.0

# Collect SQL operator durations for each operator found in CPU and GPU
for app_key in cpu_stage_log:
    for op_key in cpu_stage_log[app_key]:
        if op_key not in cpu_stage_durations:
            cpu_stage_durations[op_key] = {}
        cpu_stage_durations[op_key][app_key] = cpu_stage_log[app_key][op_key]
        cpu_stage_total = cpu_stage_total + cpu_stage_log[app_key][op_key]

for app_key in gpu_stage_log:
    for op_key in gpu_stage_log[app_key]:
        if op_key not in gpu_stage_durations:
            gpu_stage_durations[op_key] = {}
        gpu_stage_durations[op_key][app_key] = gpu_stage_log[app_key][op_key]
        gpu_stage_total = gpu_stage_total + gpu_stage_log[app_key][op_key]


def regress(op, cpu_durations, gpu_durations):
    common_keys = sorted(cpu_durations.keys() & gpu_durations.keys())
    assert len(common_keys) > 0, "No common keys found"
    x = [cpu_durations[k] for k in common_keys]
    y = [gpu_durations[k] for k in common_keys]
    plt.scatter(x, y)

    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    if len(common_keys) <= 3:
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        baseline = model.intercept_
        speedup = 1.0 / model.coef_[0]
    else:
        model = RANSACRegressor()
        model.fit(x, y)
        baseline = model.estimator_.intercept_
        speedup = 1.0 / model.estimator_.coef_[0]

    regression_line = baseline + x / speedup
    plt.plot(x, regression_line, color='red')
    plt.xlabel('CPU Runtime')
    plt.ylabel('GPU Runtime')
    plt.title(f'{op} (baseline={baseline:.2f}, speedup={speedup:.2f})')
    plt.savefig(f'{op}.png', dpi=300)
    plt.close()

    return str(round(baseline, 2)), str(round(speedup, 2))


# Create dictionary of execs where speedup factors can be calculated
scores_dict = {}

# Scan operators
if 'Scan parquet ' in cpu_stage_durations and 'GpuScan parquet ' in gpu_stage_durations:
    scores_dict["BatchScanExec"] = regress("Scan parquet", cpu_stage_durations['Scan parquet '], gpu_stage_durations['GpuScan parquet '])
    scores_dict["FileSourceScanExec"] = scores_dict["BatchScanExec"] 
if 'Scan orc ' in cpu_stage_durations and 'GpuScan orc ' in gpu_stage_durations:
    scores_dict["BatchScanExec"] = regress("Scan orc", cpu_stage_durations['Scan orc '], gpu_stage_durations['GpuScan orc '])
    scores_dict["FileSourceScanExec"] = scores_dict["BatchScanExec"]

# Other operators
if 'Expand' in cpu_stage_durations and 'GpuExpand' in gpu_stage_durations:
    scores_dict["ExpandExec"] = regress("Expand", cpu_stage_durations['Expand'], gpu_stage_durations['GpuExpand'])
if 'CartesianProduct' in cpu_stage_durations and 'GpuCartesianProduct' in gpu_stage_durations:
    scores_dict["CartesianProductExec"] = regress("CartesianProduct", cpu_stage_durations['CartesianProduct'], gpu_stage_durations['GpuCartesianProduct'])
if 'Filter' in cpu_stage_durations and 'GpuFilter' in gpu_stage_durations:
    scores_dict["FilterExec"] = regress("Filter", cpu_stage_durations['Filter'], gpu_stage_durations['GpuFilter'])
if 'SortMergeJoin' in cpu_stage_durations and 'GpuShuffledHashJoin' in gpu_stage_durations:
    scores_dict["SortMergeJoinExec"] = regress("SortMergeJoin", cpu_stage_durations['SortMergeJoin'], gpu_stage_durations['GpuShuffledHashJoin'])
if 'BroadcastHashJoin' in cpu_stage_durations and 'GpuBroadcastHashJoin' in gpu_stage_durations:
    scores_dict["BroadcastHashJoinExec"] = regress("BroadcastHashJoin", cpu_stage_durations['BroadcastHashJoin'], gpu_stage_durations['GpuBroadcastHashJoin'])
if 'Exchange' in cpu_stage_durations and 'GpuColumnarExchange' in gpu_stage_durations:
    scores_dict["ShuffleExchangeExec"] = regress("Exchange", cpu_stage_durations['Exchange'], gpu_stage_durations['GpuColumnarExchange'])
if 'HashAggregate' in cpu_stage_durations and 'GpuHashAggregate' in gpu_stage_durations:
    scores_dict["HashAggregateExec"] = regress("HashAggregate", cpu_stage_durations['HashAggregate'], gpu_stage_durations['GpuHashAggregate'])
    scores_dict["ObjectHashAggregateExec"] = scores_dict["HashAggregateExec"]
    scores_dict["SortAggregateExec"] = scores_dict["HashAggregateExec"]
if 'TakeOrderedAndProject' in cpu_stage_durations and 'GpuTopN' in gpu_stage_durations:
    scores_dict["TakeOrderedAndProjectExec"] = regress("TakeOrderedAndProject", cpu_stage_durations['TakeOrderedAndProject'], gpu_stage_durations['GpuTopN'])
if 'BroadcastNestedLoopJoin' in cpu_stage_durations and 'GpuBroadcastNestedLoopJoin' in gpu_stage_durations:
    scores_dict["BroadcastNestedLoopJoinExec"] = regress("BroadcastNestedLoopJoin", cpu_stage_durations['BroadcastNestedLoopJoin'], gpu_stage_durations['GpuBroadcastNestedLoopJoin'])

# Set overall speedup for default value for execs not in logs
overall_baseline = "0"
overall_speedup = str(max(min_speedup, round(cpu_duration/gpu_duration, 2)))

# Print out node metrics (if verbose)
if verbose:
    print("# CPU Operator Metrics")
    for key in cpu_stage_durations:
        print(key + " = " + str(cpu_stage_durations[key]))
    print("# GPU Operator Metrics")
    for key in gpu_stage_durations:
        print(key + " = " + str(gpu_stage_durations[key]))
    print("# Summary Metrics")
    print("CPU Total = " + str(cpu_stage_total))
    print("GPU Total = " + str(gpu_stage_total))
    print("Overall speedup = " + overall_speedup)

    # Print out individual exec speedup factors
    print("# Baseline and Speedup Factors ")
    for key in scores_dict:
        print(f"{key} = {scores_dict[key]}")

# Load in list of operators and set initial values to default speedup
scores_df = pd.read_csv("operatorsList.csv")
scores_df["Baseline"] = overall_baseline
scores_df["Score"] = overall_speedup

# Update operators that are found in benchmark
for key in scores_dict:
    scores_df.loc[scores_df['CPUOperator'] == key, 'Baseline'] = scores_dict[key][0]
    scores_df.loc[scores_df['CPUOperator'] == key, 'Score'] = scores_dict[key][1]

# Add in hard-coded defaults
defaults_df = pd.read_csv("defaultScores.csv")

# Generate output CSV file
final_df = pd.concat([scores_df, defaults_df])
final_df.to_csv(output, index=False)
