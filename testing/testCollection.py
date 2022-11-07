import multiprocessing
import os
import re
import time
from typing import List
from box import Box
from configs import log
from dataCollector.OfflineProfilingDataCollector import OfflineProfilingDataCollector
from deployment.deployer import Deployer
from workloadGenerator.staticWorkload import StaticWorkloadGenerator
from infGenerator.busyInf import BusyInf
from utils.others import parse_mem

import configs

DEPLOYER = Deployer(
    configs.GLOBAL_CONFIG.namespace,
    configs.GLOBAL_CONFIG.pod_spec.cpu_size,
    configs.GLOBAL_CONFIG.pod_spec.mem_size,
    configs.GLOBAL_CONFIG.nodes_for_test,
    configs.GLOBAL_CONFIG.yaml_repo_path,
    configs.GLOBAL_CONFIG.app_img,
)


def full_init(app, port):
    os.system(f"mkdir tmp")
    os.system(f"mkdir log")
    DEPLOYER.full_init(app, configs.GLOBAL_CONFIG.nodes_for_infra, port)


def init_app(containers=None):
    os.system(f"mkdir tmp")
    os.system(f"mkdir log")
    # Initiallizing corresponding APP
    DEPLOYER.redeploy_app(containers)


def start_test(continues=False):
    """Test and collect data under different <cpu, memory> interference pair.
    To adjust test parameters, please check these configuration files:
    """
    os.system(f"mkdir tmp")
    os.system(f"mkdir log")
    if not continues:
        os.system(f"rm -rf {configs.GLOBAL_CONFIG.data_path}")
        os.system(f"mkdir -p {configs.GLOBAL_CONFIG.data_path}")
    # Prepare interference generator
    interference_duration = str(
        configs.TESTING_CONFIG.duration
        * max(
            [
                x[1].max_clients
                for x in configs.TESTING_CONFIG.workload_config.services.items()
            ]
        )
        * 2
    )
    cpuInterGenerator = BusyInf(
        configs.GLOBAL_CONFIG.nodes_for_test,
        configs.TESTING_CONFIG.interference_config.cpu.cpu_size,
        configs.TESTING_CONFIG.interference_config.cpu.mem_size,
        "cpu",
        [f"{interference_duration}s"],
    )
    memoryInterGenerator = BusyInf(
        configs.GLOBAL_CONFIG.nodes_for_test,
        configs.TESTING_CONFIG.interference_config.mem.cpu_size,
        configs.TESTING_CONFIG.interference_config.mem.mem_size,
        "memory",
        [f"{interference_duration}s", "wired", "100000s"],
    )

    # Prepare data collector
    dataCollector = OfflineProfilingDataCollector(
        configs.GLOBAL_CONFIG.namespace,
        configs.TESTING_CONFIG.collector_config.jaeger_host,
        configs.TESTING_CONFIG.collector_config.entry_point,
        configs.GLOBAL_CONFIG.prometheus_host,
        configs.GLOBAL_CONFIG.nodes_for_test,
        configs.GLOBAL_CONFIG.data_path,
        max_traces=configs.TESTING_CONFIG.collector_config.max_traces,
        mointorInterval=configs.TESTING_CONFIG.collector_config.monitor_interval,
        duration=configs.TESTING_CONFIG.duration,
    )

    # Time estimation
    totalRound = (
        len(configs.TESTING_CONFIG.repeats)
        * sum(
            [
                x[1].max_clients
                for x in configs.TESTING_CONFIG.workload_config.services.items()
                if x[0] in configs.TESTING_CONFIG.services
            ]
        )
        * len(configs.TESTING_CONFIG.interference_config.cpu.pod_range)
        * len(configs.TESTING_CONFIG.interference_config.mem.pod_range)
    )
    passedRound = 0
    usedTime = 0
    # Test each service in the configuration
    for service in configs.TESTING_CONFIG.services:
        # Prepare service's needed containers
        containers = configs.GLOBAL_CONFIG.replicas[service]
        # Prepare workload generator
        currentWorkloadConfig = configs.TESTING_CONFIG.workload_config.services[service]
        workloadGenerator = StaticWorkloadGenerator(
            currentWorkloadConfig.thread_num,
            currentWorkloadConfig.connection_num,
            configs.TESTING_CONFIG.duration,
            currentWorkloadConfig.throughput,
            configs.TESTING_CONFIG.workload_config.wrk_path,
            currentWorkloadConfig.script_path,
            currentWorkloadConfig.url,
        )
        for repeat in configs.TESTING_CONFIG.repeats:
            for cpuInstance in configs.TESTING_CONFIG.interference_config.cpu.pod_range:
                for (
                    memoryInstance
                ) in configs.TESTING_CONFIG.interference_config.mem.pod_range:
                    # Clear all previous interference and generate new interference
                    DEPLOYER.delete_app()
                    log.info("Deploying Interference...")
                    cpuInterGenerator.clearAllInterference()
                    cpuInterGenerator.generateInterference(cpuInstance, True)
                    memoryInterGenerator.clearAllInterference()
                    memoryInterGenerator.generateInterference(memoryInstance, True)
                    log.info("Deploying Application...")
                    DEPLOYER.deploy_app(containers)
                    dataCollector.wait_until_done()
                    for clientNum in range(1, currentWorkloadConfig.max_clients + 1):
                        roundStartTime = time.time()
                        log.info(
                            f"Repeat {repeat} of {service}: {clientNum} clients, {cpuInstance} CPU interference and {memoryInstance} memory interference"
                        )
                        if passedRound != 0:
                            avgTime = usedTime / passedRound
                            log.info(
                                f"Used time: {timeParser(usedTime)}, "
                                f"Avg. round time: {timeParser(avgTime)}, "
                                f"Left time estimation: {timeParser(avgTime * (totalRound - passedRound))}"
                            )
                        testName = f"[{service}]{repeat}r{clientNum}c[{cpuInstance}u,{memoryInstance}m]"
                        log.info("Test starts")
                        # Start generating workload
                        startTime = int(time.time())
                        test_data = {
                            "repeat": repeat,
                            "start_time": startTime,
                            "service": service,
                            "cpu_inter": cpuInstance
                            * configs.TESTING_CONFIG.interference_config.cpu.cpu_size,
                            "mem_inter": memoryInstance
                            * parse_mem(
                                configs.TESTING_CONFIG.interference_config.mem.mem_size
                            ),
                            "target_throughput": clientNum
                            * currentWorkloadConfig.throughput,
                            "test_name": testName,
                        }
                        workloadGenerator.generateWorkload(testName, clientNum)
                        # Record test result data
                        dataCollector.collect_data_async(test_data)
                        passedRound += 1
                        usedTime += time.time() - roundStartTime
    dataCollector.wait_until_done()


def timeParser(time):
    time = int(time)
    hours = format(int(time / 3600), "02d")
    minutes = format(int((time % 3600) / 60), "02d")
    secs = format(int(time % 3600 % 60), "02d")
    return f"{hours}:{minutes}:{secs}"
