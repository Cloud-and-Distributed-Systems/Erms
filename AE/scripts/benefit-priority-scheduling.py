import argparse
import os
from AE.scripts.utils import (
    CONFIG,
    container_computation,
    get_inter,
)
from utils.files import append_data
import pandas as pd


def sharing_main(
    data_path,
    result_path,
    inf,
    slas_workloads,
    services,
):
    result_file = f"{result_path}/{'_'.join(services)}.csv"
    print(f"Data will be saved into {result_file}")
    latency_target_data, container_data = container_computation(
        data_path,
        slas_workloads,
        inf["cpu"],
        inf["mem"],
        services,
        ["erms"],
        debug=False,
        prio=True,
    )
    container_result = statistics(container_data, latency_target_data, True)
    append_data(container_result, result_file)
    latency_target_data, container_data = container_computation(
        data_path,
        slas_workloads,
        inf["cpu"],
        inf["mem"],
        services,
        ["erms"],
        debug=False,
        prio=False,
    )
    container_result = statistics(container_data, latency_target_data, False)
    append_data(container_result, result_file)
    print(f"Done.")


def statistics(container_data: pd.DataFrame, latency_target_data: pd.DataFrame, prio):
    container_data = container_data.loc[
        container_data["microservice"] != "nginx-web-server"
    ]
    columns = ["service", "cpu_inter", "mem_inter", "var_index", "method"]
    data = (
        latency_target_data.groupby(columns + ["microservice"])["container"]
        .max()
        .groupby(columns)
        .sum()
        .reset_index()
    )
    services = latency_target_data["service"].unique().tolist()
    service_1 = services[0]
    latency_target_1 = (
        data.loc[data["service"] == service_1]
        .merge(
            latency_target_data[
                ["service", "var_index", f"{service_1}_sla", f"{service_1}_workload"]
            ].drop_duplicates(),
            on=["service", "var_index"],
        )
        .rename(
            columns={f"{service_1}_sla": "sla_1", f"{service_1}_workload": "workload_1"}
        )
    )
    if len(services) > 1:
        service_2 = services[1]
        latency_target_2 = (
            data.loc[data["service"] == service_2]
            .merge(
                latency_target_data[
                    [
                        "service",
                        "var_index",
                        f"{service_2}_sla",
                        f"{service_2}_workload",
                    ]
                ].drop_duplicates(),
                on=["service", "var_index"],
            )
            .rename(
                columns={
                    f"{service_2}_sla": "sla_2",
                    f"{service_2}_workload": "workload_2",
                }
            )
        )
        container_merge = (
            container_data.groupby(["cpu_inter", "mem_inter", "method", "var_index"])[
                "container"
            ]
            .sum()
            .reset_index()
            .rename(columns={"container": "container_merge"})
        )
        result = latency_target_1.merge(
            latency_target_2,
            on=["cpu_inter", "mem_inter", "var_index", "method"],
            suffixes=("_1", "_2"),
        ).merge(container_merge, on=["cpu_inter", "mem_inter", "var_index", "method"])
    else:
        result = latency_target_1.assign(
            service_2=None,
            container_2=None,
            sla_2=None,
            workload_2=None,
            container_merge=latency_target_1["container"],
        ).rename(columns={"container": "container_1", "service": "service_1"})
    result = result.assign(use_prio=prio, latency_1=None, latency_2=None).rename(
        columns={"cpu_inter": "cpu_inf", "mem_inter": "mem_inf"}
    )
    result = result[
        [
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


parser = argparse.ArgumentParser(
    description=(
        "This script will perform latency target computation and resource allocation "
        "with and without priority scheduling using Erms."
        "Ususally, this script will consume 5 minutes for each application. "
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
args = parser.parse_args()
input_path = args.profiling_data
output_path = args.directory

os.system(f"rm -rf {output_path}/benefit-priority-scheduling")
for app in ["hotel-reserv", "social-network"]:
    data_path = f"{input_path}/data_{app}"
    services = {
        "hotel-reserv": ["Search", "Recommendation"],
        "social-network": ["ComposePost", "UserTimeline"],
    }[app]
    slas_workloads = {
        "social-network": [
            ({"ComposePost": a, "UserTimeline": b}, {"ComposePost": c, "UserTimeline": d})
            for a in [300, 350, 400]
            for b in [200, 250, 300]
            for c in [140, 210]
            for d in [120, 180]
        ],
        "hotel-reserv": [
            ({"Search": a, "Recommendation": b}, {"Search": c, "Recommendation": d})
            for a in [200, 300, 400]
            for b in [50, 70, 90]
            for c in [60, 180]
            for d in [140, 420]
        ],
    }[app]

    cpu_inf, mem_inf = get_inter(CONFIG["interference"][1])
    inf = {"cpu": cpu_inf, "mem": mem_inf}

    os.system(f"mkdir -p {output_path}/benefit-priority-scheduling")
    sharing_main(
        data_path,
        f"{output_path}/benefit-priority-scheduling",
        inf,
        slas_workloads,
        services,
    )
