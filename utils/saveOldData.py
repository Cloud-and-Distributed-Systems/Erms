from datetime import datetime
import os
import re
from typing import Dict, List, Set
import pandas as pd
import utils.timer as timer
import utils.traceProcessor as t_processor
from utils.files import append_data


def read_data(data_path, log_path):
    data = pd.read_csv(f"{data_path}/offlineTestResult/originalData.csv")
    services = data["service"].unique().tolist()
    logs = {}
    for service in services:
        with open(f"{log_path}/{service}.log") as file:
            logs[service] = file.readlines()
    return data, logs


def time_to_interference(
    logs: Dict, cpu_inter_range, mem_inter_range, max_clients_dict
):
    from_time = 0
    to_time = 0
    result = pd.DataFrame(
        columns=[
            "service",
            "repeat",
            "cpuInter",
            "memInter",
            "realThroughput",
            "targetThroughput",
            "from",
            "to",
        ]
    )
    for service, log in logs.items():
        repeat = 0
        mem_index = 0
        cpu_index = 0
        client_index = 1
        mem_length = len(mem_inter_range)
        cpu_length = len(cpu_inter_range)
        max_clients = max_clients_dict[service]
        for line in log:
            search = re.search(
                (
                    "(\d{4}\/\d{2}\/\d{2} \d{2}:\d{2}:\d{2}).*"
                    "Real Throughput: ([0-9\.]*), "
                    "Target Throughput: ([0-9\.]*)"
                ),
                line,
            )
            if search:
                date_str = search.groups()[0]
                timestamp = datetime.timestamp(
                    datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
                )
                to_time = timestamp
                result = pd.concat(
                    [
                        result,
                        pd.DataFrame(
                            {
                                "service": service,
                                "repeat": repeat,
                                "cpuInter": cpu_inter_range[cpu_index],
                                "memInter": mem_inter_range[mem_index],
                                "from": from_time,
                                "to": to_time,
                                "realThroughput": search.groups()[1],
                                "targetThroughput": search.groups()[2],
                            },
                            index=[1],
                        ),
                    ]
                )
                from_time = to_time
                client_index += 1
                if client_index > max_clients:
                    client_index = 1
                    mem_index += 1
                    if mem_index == mem_length:
                        mem_index = 0
                        cpu_index += 1
                        if cpu_index == cpu_length:
                            cpu_index = 0
                            repeat += 1
            else:
                continue
    return result


def classify_data(data: pd.DataFrame, classify_df: pd.DataFrame, file_path):
    counter = 0
    print("classify starts")

    def classify(x: pd.Series):
        target_data = data.loc[
            (x["from"] <= data["traceTime"]) & (data["traceTime"] < x["to"])
        ]
        additional_data = pd.DataFrame(
            data=[x.values] * len(target_data), columns=x.index, index=target_data.index
        )[
            [
                "repeat",
                "cpuInter",
                "memInter",
                "realThroughput",
                "targetThroughput",
            ]
        ]
        target_data = target_data[
            [
                "traceId",
                "traceTime",
                "startTime",
                "endTime",
                "parentId",
                "childId",
                "childOperation",
                "parentOperation",
                "childMS",
                "childPod",
                "parentMS",
                "parentPod",
                "parentDuration",
                "childDuration",
                "service",
            ]
        ].merge(additional_data, left_index=True, right_index=True)
        nonlocal counter
        append_data(target_data, file_path)
        counter += 1
        if counter % 10 == 0:
            print(counter)

    classify_df.apply(classify, axis=1)


groups_num = 0
processed_groups = 0


def process_data(data: pd.DataFrame, file_path):
    groupby_result = data.groupby(
        ["repeat", "cpuInter", "memInter", "realThroughput", "service"]
    )
    global groups_num
    groups_num = len(groupby_result.groups)
    groupby_result.apply(process_groupby_data, file_path=file_path)


def process_groupby_data(data: pd.DataFrame, file_path):
    global processed_groups, groups_num
    print(f"{processed_groups}/{groups_num}")
    timer.start()
    repeat, cpuInter, memInter, realThroughput, targetThroughput, service = data[
        [
            "repeat",
            "cpuInter",
            "memInter",
            "realThroughput",
            "targetThroughput",
            "service",
        ]
    ].apply(lambda x: x.iloc[0])
    data = t_processor.remove_repeatation(data)
    data = t_processor.exact_parent_duration(data, "merge")
    data = t_processor.decouple_parent_and_child(data)
    data = data.assign(
        repeat=repeat,
        cpuInter=cpuInter,
        memInter=memInter,
        reqFreq=realThroughput,
        targetReqFreq=targetThroughput,
        service=service,
    )
    append_data(data, file_path)
    timer.print_duration(f"group {processed_groups}")
    processed_groups += 1


def remove_client_rt(data: pd.DataFrame, file_path):
    before = len(data)
    data = data.loc[data["childOperation"].str.find("ClientRT") == -1]
    after = len(data)
    print(before - after)
    append_data(data, file_path)
    return data
