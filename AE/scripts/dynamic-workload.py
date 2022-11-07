import argparse
import os
import pandas as pd
from dataCollector.OfflineProfilingDataCollector import OfflineProfilingDataCollector
from AE.scripts.utils import CONFIG
from copy import deepcopy
from onlineScheduling.placementOptimization import ErmsScheduler
from AE.scripts.utils import (
    config_to_workload,
    container_computation,
    deploy_infs,
    get_inter,
    k8s_scheduling,
)

from workloadGenerator.dynamicWorkload import DynamicWorkloadGenerator
from deployment.deployer import delete_by_yaml
from time import time

parser = argparse.ArgumentParser(
    description=(
        "This script will perform dynamic workload test with different resource allocation "
        "methods."
        "Ususally, this script will consume 20 hours. "
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--dir",
    "-d",
    dest="directory",
    help="Directory to store data and figures",
    default="AE/data"
)
parser.add_argument(
    "--profiling-data",
    "-p",
    dest="profiling_data",
    help="Path to the profiling data (if AECs use their own profiling data)",
    default="AE/data",
)
parser.add_argument(
    "--repeats", "-r", dest="repeats", help="Repeats of test", default=5
)
args = parser.parse_args()
output_path = args.directory
input_path = args.profiling_data
repeats = int(args.repeats)


sla = 200
scale = 0.3
service = "ComposePost"
data_path = f"data_social"


def main(
    interference,
    sla,
    workload_config,
    scale,
    data_path,
    service,
    nodes,
    namespace,
    pod_spec,
    yaml_repo,
    image,
    prometheus_host,
    script,
    host,
    operation,
    repeats,
    result_path,
    jaeger_host,
    entry_point,
    methods=["erms"],
    no_nginx=False,
    no_frontend=False,
    do_inf_deployment=False,
    init_containers=None,
    fixed_ms=None,
    base_workload=None,
):
    collector = OfflineProfilingDataCollector(
        namespace, jaeger_host, entry_point, prometheus_host, nodes, data_path
    )
    generator = DynamicWorkloadGenerator("wrk2/wrk", script, host)
    scheduler = ErmsScheduler(data_path)
    # Generate dynamic workload estimation
    clients_list = generator.workload_sequence(scale)
    print(clients_list)
    slas_workload_configs = []
    for clients in clients_list:
        updated_workload_config = deepcopy(workload_config)
        updated_workload_config["clients"] = clients
        slas_workload_configs.append(
            ({service: sla}, {service: updated_workload_config})
        )
    slas_workloads = config_to_workload(slas_workload_configs)
    # Latency target computation
    print("Computing latency targets...")
    cpu_inter, mem_inter = get_inter(interference)
    latency_targets, containers = container_computation(
        data_path,
        slas_workloads,
        cpu_inter,
        mem_inter,
        [service],
        methods,
        init_containers=init_containers,
        base_workload=base_workload,
        fixed_ms=fixed_ms,
    )
    latency_targets = latency_targets.rename(columns={"var_index": "timestamp"})
    containers = containers.rename(columns={"var_index": "timestamp"})
    # Deploy interference
    if do_inf_deployment:
        print("Deploying interference...")
        deploy_infs(interference, nodes)
    for method in methods:
        for repeat in repeats:
            method_containers = containers.loc[containers["method"] == method]
            gen_procs = generator.generate_workload(slas_workload_configs, service)
            for index, (_, grp) in enumerate(method_containers.groupby("timestamp")):
                grp.at[
                    grp.loc[grp["microservice"] == "compose-post-service"].index[0],
                    "container",
                ] += 3
                grp.at[
                    grp.loc[grp["microservice"] == "url-shorten-service"].index[0],
                    "container",
                ] += 2
                print(grp)
                if method == "erms":
                    scheduler.main(
                        pod_spec,
                        1,
                        nodes,
                        yaml_repo,
                        namespace,
                        prometheus_host,
                        image,
                        latency_target_df=latency_targets.loc[
                            latency_targets["timestamp"] == index
                        ],
                        container_df=grp,
                    )
                else:
                    k8s_scheduling(
                        grp, nodes, namespace, pod_spec, yaml_repo, image, "tmp/dynamic"
                    )
                start_time = time()
                print(
                    f"Timestamp {index}, workload {slas_workloads[index][1][service]}, testing..."
                )
                gen_procs[index].start()
                gen_procs[index].join()
                collector.validation_collection_async(
                    f"dynamic_{index}",
                    start_time,
                    operation,
                    service,
                    repeat,
                    f"{result_path}/{service}",
                    no_nginx,
                    no_frontend,
                    method=method,
                    sla=sla,
                    workload=slas_workloads[index][1][service],
                    timestamp=index,
                    container=grp["container"].sum(),
                )
            collector.wait_until_done()
    delete_by_yaml("tmp/dynamic")


os.system(f"mv {data_path}/dynamic-workload {data_path}/dynamic-workload_original")
os.system(f"rm -rf {output_path}/dynamic-workload")
os.system(f"mkdir -p {output_path}/dynamic-workload")
app = "social-network"
main(
    CONFIG["interference"][1],
    sla,
    CONFIG["workload_config"][app],
    scale,
    data_path,
    service,
    CONFIG["nodes"],
    CONFIG["namespace"][app],
    CONFIG["pod_spec"],
    CONFIG["yaml_repo"][app],
    CONFIG["image"][app],
    CONFIG["prometheus_host"],
    CONFIG["scripts"][app][service],
    CONFIG["host"][app],
    CONFIG["operations"][app][service],
    range(repeats),
    "dynamic-workload",
    CONFIG["jaeger_host"][app],
    CONFIG["entry_point"][app],
    no_nginx=True,
    init_containers=CONFIG["init_containers"][app],
    base_workload=CONFIG["base_workload"][app],
    fixed_ms=CONFIG["fixed_ms"][app],
    do_inf_deployment=True,
)
os.system(f"mv {data_path}/dynamic-workload/* {output_path}/dynamic-workload")
os.system(f"rmdir dynamic-workload")
os.system(f"mv {data_path}/dynamic-workload_original {data_path}/dynamic-workload")

file = f"{output_path}/dynamic-workload/{service}/trace_latency.csv"
data = pd.read_csv(file)
columns = ["service", "method", "timestamp"]
data = (
    data.groupby(columns + ["repeat"])
    .quantile(0.95)
    .groupby(columns)
    .mean()
    .reset_index()
)
data = data.assign(sla=data["sla"] * 1000)
data.to_csv(f"{output_path}/dynamic-workload/{app}.csv", index=False)
