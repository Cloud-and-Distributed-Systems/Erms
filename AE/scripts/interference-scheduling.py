import argparse
import os
from time import sleep
import pandas as pd
from dataCollector.OfflineProfilingDataCollector import OfflineProfilingDataCollector
from deployment.deployer import delete_by_yaml
from onlineScheduling.applyPriority import clear_vifs
from onlineScheduling.placementOptimization import ErmsScheduler
from AE.scripts.utils import (
    config_to_workload,
    container_computation,
    deploy_infs,
    get_inter,
    k8s_scheduling,
    generate_multiple_workloads,
)
from math import ceil
from AE.scripts.utils import CONFIG


def figure_a(
    inters,
    slas_workload_configs,
    data_path,
    service,
    nodes,
    namespace,
    pod_spec,
    yaml_repo,
    image,
    prometheus_host,
    scripts,
    host,
    operations,
    repeats,
    result_path,
    jaeger_host,
    entry_point,
    ratios,
    no_nginx=False,
    no_frontend=False,
):
    collector = OfflineProfilingDataCollector(
        namespace, jaeger_host, entry_point, prometheus_host, nodes, data_path
    )
    # Latency target computation
    for inter in inters:
        cpu_inter, mem_inter = get_inter(inter)
        print("Computing latency targets...")
        slas_workloads = config_to_workload(slas_workload_configs)
        _, containers_result = container_computation(
            data_path, slas_workloads, cpu_inter, mem_inter, [service], ["erms"]
        )
        print("Deploying interference...")
        deploy_infs(inter, nodes)
        for var_index, (sla, workload_config) in enumerate(slas_workload_configs):
            for repeat in repeats:
                for ratio in ratios:
                    containers = containers_result.loc[
                        containers_result["var_index"] == var_index
                    ]
                    entrance_index = containers["microservice"].isin(
                        ["frontend", "nginx", "nginx-web-server"]
                    )
                    entrance = containers.loc[entrance_index]
                    others = containers.loc[~entrance_index]
                    original_containers = others["container"].sum()
                    others = others.assign(
                        container=(others["container"] * ratio).apply(ceil)
                    )
                    print(f"Ratio: {others['container'].sum() / original_containers}")
                    containers = pd.concat([entrance, others])
                    print(containers)
                    # Containers deployment
                    print("Deploying containers...")
                    k8s_scheduling(
                        containers, nodes, namespace, pod_spec, yaml_repo, image
                    )
                    if no_frontend:
                        sleep(20)
                    print("Test starts...")
                    # Workload generation
                    start_time = generate_multiple_workloads(
                        [service], 60, workload_config, scripts, host
                    )
                    # Data collection
                    kwargs = {
                        f"{service}_sla": sla[service] * 1000,
                        f"{service}_workload": slas_workloads[var_index][1][service],
                        "container": others["container"].sum(),
                        "original_container": original_containers,
                        "var_index": var_index,
                        "cpu_inter": cpu_inter,
                        "mem_inter": mem_inter,
                    }
                    collector.validation_collection_async(
                        f"validation_{service}",
                        start_time,
                        operations[service],
                        service,
                        repeat,
                        result_path,
                        no_nginx,
                        no_frontend,
                        **kwargs,
                    )
    print("Waiting for data collection...")
    collector.wait_until_done()
    delete_by_yaml("tmp/scheduledAPP")


def figure_b(
    inters,
    slas_workload_configs,
    data_path,
    service,
    nodes,
    namespace,
    pod_spec,
    yaml_repo,
    image,
    prometheus_host,
    scripts,
    host,
    operations,
    repeats,
    result_path,
    jaeger_host,
    entry_point,
    no_nginx=False,
    no_frontend=False,
):
    erms_scheduling = ErmsScheduler(data_path)
    collector = OfflineProfilingDataCollector(
        namespace, jaeger_host, entry_point, prometheus_host, nodes, data_path
    )
    # Latency target computation
    for inter in inters:
        cpu_inter, mem_inter = get_inter(inter)
        print("Deploying interference...")
        deploy_infs(inter, nodes)
        print("Computing latency targets...")
        slas_workloads = config_to_workload(slas_workload_configs)
        latency_targets_result, containers_result = container_computation(
            data_path, slas_workloads, cpu_inter, mem_inter, [service], ["erms"]
        )
        for var_index, (sla, workload_config) in enumerate(slas_workload_configs):
            for repeat in repeats:
                for scheduler in ["erms", "k8s"]:
                    containers = containers_result.loc[
                        containers_result["var_index"] == var_index
                    ]
                    latency_targets = latency_targets_result.loc[
                        latency_targets_result["var_index"] == var_index
                    ]
                    print(containers)
                    # Containers deployment
                    print("Deploying containers...")
                    if scheduler == "erms":
                        erms_scheduling.main(
                            pod_spec,
                            1,
                            nodes,
                            yaml_repo,
                            namespace,
                            prometheus_host,
                            image,
                            latency_target_df=latency_targets,
                            container_df=containers,
                        )
                    elif scheduler == "k8s":
                        k8s_scheduling(
                            containers, nodes, namespace, pod_spec, yaml_repo, image
                        )
                    else:
                        continue
                    if no_frontend:
                        sleep(30)
                    print("Test starts...")
                    # Workload generation
                    start_time = generate_multiple_workloads(
                        [service], 60, workload_config, scripts, host
                    )
                    # Data collection
                    kwargs = {
                        f"{service}_sla": sla[service] * 1000,
                        f"{service}_workload": slas_workloads[var_index][1][service],
                        "scheduler": scheduler,
                        "var_index": var_index,
                        "cpu_inter": cpu_inter,
                        "mem_inter": mem_inter,
                    }
                    collector.validation_collection_async(
                        f"validation_{service}",
                        start_time,
                        operations[service],
                        service,
                        repeat,
                        result_path,
                        no_nginx,
                        no_frontend,
                        **kwargs,
                    )
                    if scheduler == "erms":
                        clear_vifs(data_path)
    print("Waiting for data collection...")
    collector.wait_until_done()
    delete_by_yaml("tmp/scheduledAPP")


def statistics_b(folder):
    data = pd.read_csv(f"{folder}/trace_latency.csv")
    columns = ["service", "var_index", "cpu_inter", "mem_inter", "scheduler"]
    data = (
        data.groupby(columns + ["repeat"])
        .quantile(0.95)
        .groupby(columns)
        .mean()
        .reset_index()
    )
    service = data["service"].unique().tolist()[0]
    data = data.rename(
        columns={f"{service}_sla": "sla", f"{service}_workload": "workload"}
    )
    return data


def statistics_a(file_path):
    data = pd.read_csv(f"{file_path}/trace_latency.csv")
    columns = ["service", "var_index", "cpu_inter", "mem_inter", "container"]
    data = (
        data.groupby(columns + ["repeat"])
        .quantile(0.95)
        .groupby(columns)
        .mean()
        .reset_index()
    )
    service = data["service"].unique().tolist()[0]
    data = data.rename(
        columns={f"{service}_sla": "sla", f"{service}_workload": "workload"}
    )
    return data


parser = argparse.ArgumentParser(
    description=(
        "This script will perform interference-based scheduling test. "
        "Ususally, this script will consume 2 hours. "
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--dir",
    "-d",
    dest="directory",
    help="Directory to store data and figures",
    default="AE/data",
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
app = "hotel-reserv"
repeats = int(args.repeats)

data_path = f"{input_path}/data_{app}"
slas_workload_configs = {
    "social-network": [
        (
            {"UserTimeline": 35},
            {"UserTimeline": {"clients": 20, "thread": 2, "conn": 3, "throughput": 6}},
        ),
        (
            {"UserTimeline": 40},
            {"UserTimeline": {"clients": 20, "thread": 2, "conn": 3, "throughput": 6}},
        ),
    ],
    "hotel-reserv": [
        (
            {"Search": 200},
            {"Search": {"clients": 20, "thread": 2, "conn": 4, "throughput": 8}},
        ),
        (
            {"Search": 300},
            {"Search": {"clients": 20, "thread": 2, "conn": 4, "throughput": 8}},
        ),
    ],
}[app]
inters = [
    {
        "cpu": {
            "cpu_size": 0.25,
            "mem_size": "10Mi",
            "allocation": {
                "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 8},
                "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 8},
                "izj6c6vb9bfm8mxnvb4n45z": {"idle": 6, "busy": 0},
                "izj6c6vb9bfm8mxnvb4n44z": {"idle": 6, "busy": 0},
            },
        },
        "mem": {
            "cpu_size": 0.01,
            "mem_size": "500Mi",
            "allocation": {
                "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 8},
                "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 8},
                "izj6c6vb9bfm8mxnvb4n45z": {"idle": 6, "busy": 0},
                "izj6c6vb9bfm8mxnvb4n44z": {"idle": 6, "busy": 0},
            },
        },
    },
    {
        "cpu": {
            "cpu_size": 0.25,
            "mem_size": "10Mi",
            "allocation": {
                "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 6},
                "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 6},
                "izj6c6vb9bfm8mxnvb4n45z": {"idle": 3, "busy": 0},
                "izj6c6vb9bfm8mxnvb4n44z": {"idle": 3, "busy": 0},
            },
        },
        "mem": {
            "cpu_size": 0.01,
            "mem_size": "500Mi",
            "allocation": {
                "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 6},
                "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 6},
                "izj6c6vb9bfm8mxnvb4n45z": {"idle": 3, "busy": 0},
                "izj6c6vb9bfm8mxnvb4n44z": {"idle": 3, "busy": 0},
            },
        },
    },
]
service = {"hotel-reserv": "Search"}[app]

os.system(f"mv {data_path}/interference-scheduling {data_path}/interference-scheduling_original")
os.system(f"rm -rf {output_path}/interference-scheduling")
os.system(f"mkdir -p {output_path}/interference-scheduling")
figure_a(
    inters,
    slas_workload_configs,
    data_path,
    service,
    CONFIG["nodes"],
    CONFIG["namespace"][app],
    CONFIG["pod_spec"],
    CONFIG["yaml_repo"][app],
    CONFIG["image"][app],
    CONFIG["prometheus_host"],
    CONFIG["scripts"][app],
    CONFIG["host"][app],
    CONFIG["operations"][app],
    range(repeats),
    f"/{service}-a",
    CONFIG["jaeger_host"][app],
    CONFIG["entry_point"][app],
    [1.2, 1.4, 1.6, 1.8],
    no_nginx=app == "social-network",
    no_frontend=app == "hotel-reserv",
)

figure_b(
    inters,
    slas_workload_configs,
    data_path,
    service,
    CONFIG["nodes"],
    CONFIG["namespace"][app],
    CONFIG["pod_spec"],
    CONFIG["yaml_repo"][app],
    CONFIG["image"][app],
    CONFIG["prometheus_host"],
    CONFIG["scripts"][app],
    CONFIG["host"][app],
    CONFIG["operations"][app],
    range(repeats),
    f"interference-scheduling/{service}-b",
    CONFIG["jaeger_host"][app],
    CONFIG["entry_point"][app],
    no_nginx=app == "social-network",
    no_frontend=app == "hotel-reserv",
)
os.system(f"mv {data_path}/interference-scheduling/* {output_path}/interference-scheduling")
os.system(f"rmdir {data_path}/interference-scheduling")
os.system(f"mv {data_path}/interference-scheduling_original {data_path}/interference-scheduling")

statistics_a(f"{output_path}/interference-scheduling/{service}-a").to_csv(
    f"{output_path}/interference-scheduling/{service}_a.csv", index=False
)
statistics_b(f"{output_path}/interference-scheduling/{service}-b").to_csv(
    f"{output_path}/interference-scheduling/{service}_b.csv", index=False
)
