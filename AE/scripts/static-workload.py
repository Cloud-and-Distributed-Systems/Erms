import argparse, os
from time import sleep
import pandas as pd
from onlineScheduling.applyPriority import clear_vifs
from onlineScheduling.placementOptimization import ErmsScheduler
from AE.scripts.utils import CONFIG
from AE.scripts.utils import config_to_workload, deploy_infs, get_inter
from deployment.deployer import delete_by_yaml
from dataCollector.OfflineProfilingDataCollector import OfflineProfilingDataCollector
from AE.scripts.utils import (
    container_computation,
    k8s_scheduling,
    generate_multiple_workloads,
)

parser = argparse.ArgumentParser(
    description=(
        "This script will perform static workload test with different resource allocation "
        "methods. "
        "Ususally, this script will consume 5 hours for each application. "
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--profiling-data",
    "-p",
    dest="profiling_data",
    help="Path to the profiling data (if AECs use their own profiling data)",
    default="AE/data",
)
parser.add_argument(
    "--dir",
    "-d",
    dest="directory",
    help="Directory to store data and figures",
    default="AE/data",
)

parser.add_argument(
    "--repeats", "-r", dest="repeats", help="Repeats of test", default=5
)
args = parser.parse_args()
input_path = args.profiling_data
output_path = args.directory
repeats = int(args.repeats)



def sharing_main(
    interference,
    slas_workload_configs,
    data_path,
    services,
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
    init_containers=None,
    base_workload=None,
    fixed_ms=None,
    no_nginx=False,
    no_frontend=False,
    do_inf_deployment=True,
    methods=["erms"],
):
    erms_scheduling = ErmsScheduler(data_path)
    collector = OfflineProfilingDataCollector(
        namespace, jaeger_host, entry_point, prometheus_host, nodes, data_path
    )
    cpu_inter, mem_inter = get_inter(interference)
    # Process workload configs
    slas_workloads = config_to_workload(slas_workload_configs)
    # Latency target computation
    print("Computing latency targets...")
    latency_targets_result, containers_result = container_computation(
        data_path,
        slas_workloads,
        cpu_inter,
        mem_inter,
        services,
        methods,
        init_containers=init_containers,
        base_workload=base_workload,
        fixed_ms=fixed_ms,
    )
    # Deploy interference
    if do_inf_deployment:
        print("Deploying interference...")
        deploy_infs(interference, nodes)
    os.system("kubectl rollout restart deployment -n hotel-reserv consul")
    for method in methods:
        for repeat in repeats:
            for var_index, (sla, workload_config) in enumerate(slas_workload_configs):
                containers = containers_result.loc[
                    (containers_result["var_index"] == var_index)
                    & (containers_result["method"] == method)
                ]
                latency_targets = latency_targets_result.loc[
                    (latency_targets_result["var_index"] == var_index)
                    & (latency_targets_result["method"] == method)
                ]
                print(
                    latency_targets[
                        [
                            "service",
                            "microservice",
                            "latency_target",
                            "container",
                            "workload",
                        ]
                    ]
                )
                # Media: movie-review-service, user-review-service
                # Social: user-timeline-service
                containers.at[
                    (
                        containers["microservice"].isin(
                            [
                                "movie-review-service",
                                "user-review-service",
                            ]
                        )
                    )
                    & (containers["container"] == 1),
                    "container",
                ] = 3
                print(containers)
                # Containers deployment
                print("Deploying containers...")
                if method == "erms":
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
                elif method in ["rhythm", "grandSLAm", "firm"]:
                    k8s_scheduling(
                        containers, nodes, namespace, pod_spec, yaml_repo, image
                    )
                else:
                    continue
                if no_frontend:
                    sleep(20)
                print("Test starts...")
                # Workload generation
                start_time = generate_multiple_workloads(
                    services, 60, workload_config, scripts, host
                )
                # Data collection
                kwargs = {f"{service}_sla": sla[service] for service in services}
                kwargs.update(
                    {
                        f"{service}_workload": slas_workloads[var_index][1][service]
                        for service in services
                    }
                )
                kwargs.update(
                    {
                        "method": method,
                        "var_index": var_index,
                        "container": containers["container"].sum(),
                        "cpu_inf": cpu_inter,
                        "mem_inf": mem_inter,
                    }
                )
                for service in services:
                    kwargs.update(
                        {
                            "service_container": latency_targets.loc[
                                latency_targets["service"] == service
                            ]
                            .groupby("microservice")["container"]
                            .max()
                            .sum()
                        }
                    )
                    collector.validation_collection_async(
                        f"validation_{service}",
                        start_time,
                        operations[service],
                        service,
                        repeat,
                        f"{result_path}/{'_'.join(services)}",
                        no_nginx,
                        no_frontend,
                        **kwargs,
                    )
                if method == "erms":
                    clear_vifs(data_path)
    print("Waiting for data collection...")
    collector.wait_until_done()
    delete_by_yaml("tmp/scheduledAPP")


def statistics(result_path):
    data = pd.read_csv(f"{result_path}/trace_latency.csv")
    data = (
        data.groupby(["service", "var_index", "method", "repeat"])
        .quantile(0.95)
        .reset_index()
    )
    services = data["service"].unique().tolist()
    service_1 = services[0]
    result_1 = data.loc[data["service"] == service_1][
        [
            "service",
            "var_index",
            "method",
            "traceLatency",
            "throughput",
            f"{service_1}_sla",
            f"{service_1}_workload",
            "container",
            "cpu_inf",
            "mem_inf",
            "service_container",
            "repeat",
        ]
    ].rename(
        columns={
            "service": "service_1",
            f"{service_1}_sla": "sla_1",
            f"{service_1}_workload": "workload_1",
            "container": "container_merge",
            "traceLatency": "latency_1",
            "service_container": "container_1",
        }
    )
    if len(services) > 1:
        service_2 = services[1]
        result_2 = data.loc[data["service"] == service_2][
            [
                "service",
                "var_index",
                "method",
                "traceLatency",
                "throughput",
                f"{service_2}_sla",
                f"{service_2}_workload",
                "service_container",
                "repeat",
            ]
        ].rename(
            columns={
                "service": "service_2",
                f"{service_2}_sla": "sla_2",
                f"{service_2}_workload": "workload_2",
                "traceLatency": "latency_2",
                "service_container": "container_2",
            }
        )
        result = result_1.merge(result_2, on=["var_index", "method", "repeat"]).assign(
            use_prio=True,
        )
    else:
        result = result_1.assign(
            service_2=None,
            sla_2=None,
            workload_2=None,
            latency_2=None,
            container_2=None,
            use_prio=True,
        )
    result = result[
        [
            "repeat",
            "service_1",
            "service_2",
            "method",
            "use_prio",
            "cpu_inf",
            "mem_inf",
            "sla_1",
            "sla_2",
            "workload_1",
            "workload_2",
            "container_1",
            "container_2",
            "container_merge",
            "latency_1",
            "latency_2",
        ]
    ]
    return result


os.system(f"rm -rf {output_path}/static-workload")
for app in ["media-microsvc", "hotel-reserv"]:
    data_path = f"{input_path}/data_{app}"
    os.system(f"mv {data_path}/static-workload {data_path}/static-workload_original")
    os.system(f"mkdir -p {data_path}/static-workload")
    os.system(f"mkdir -p {output_path}/static-workload")
    interference = {
        "hotel-reserv": CONFIG["interference"][0],
        "media-microsvc": CONFIG["interference"][1],
    }[app]
    sharing_main(
        interference,
        CONFIG["slas_workload_configs"][app],
        data_path,
        CONFIG["services"][app],
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
        "static-workload",
        CONFIG["jaeger_host"][app],
        CONFIG["entry_point"][app],
        no_frontend=app == "hotel-reserv",
        no_nginx=app == "social-network",
        fixed_ms=CONFIG["fixed_ms"][app],
        init_containers=CONFIG["init_containers"][app],
        base_workload=CONFIG["base_workload"][app],
    )
    os.system(f"mv {data_path}/static-workload/* {output_path}/static-workload")
    os.system(f"rmdir {data_path}/static-workload")
    os.system(f"mv {data_path}/static-workload_original {data_path}/static-workload")

    result = statistics(f"{output_path}/static-workload/{'_'.join(CONFIG['services'][app])}")
    result.to_csv(f"{output_path}/static-workload/{app}.csv", index=False)
