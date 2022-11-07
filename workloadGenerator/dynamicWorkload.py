from math import ceil
import multiprocessing
from typing import List
import pandas as pd
from workloadGenerator.staticWorkload import StaticWorkloadGenerator


class DynamicWorkloadGenerator:
    def __init__(self, wrk_path, script_path, url):
        self.wrk_path = wrk_path
        self.script_path = script_path
        self.url = url
        self.processes = []
        pass

    @staticmethod
    def workload_sequence(scale: float):
        cpm = pd.read_csv("workloadGenerator/dynamicWorkload.csv")
        return (cpm["tps_truth"] * scale).apply(ceil).tolist()

    def generate_workload(self, slas_workload_configs, service, test_duration=60):
        processes: List[multiprocessing.Process] = []
        for index, (_, workload_config) in enumerate(slas_workload_configs):
            static_generator = StaticWorkloadGenerator(
                workload_config[service]["thread"],
                workload_config[service]["conn"],
                test_duration,
                workload_config[service]["throughput"],
                self.wrk_path,
                self.script_path,
                self.url,
            )
            process = multiprocessing.Process(
                target=static_generator.generateWorkload,
                args=(f"dynamic_{index}", workload_config[service]["clients"]),
            )
            processes.append(process)
        return processes
