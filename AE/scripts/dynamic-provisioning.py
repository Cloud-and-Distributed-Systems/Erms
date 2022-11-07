import os
import pandas as pd
from box import Box
from utils.others import parse_mem
from onlineScheduling.placementOptimization import ErmsScheduler
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online scaling, latency target computation part",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--app",
        "-a",
        dest="app",
        help="Application name. (`hotel-reserv`, `social-network`, `media-microsvc`)",
        default="hotel-reserv",
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        help="Config file path.",
        default="AE/scripts/configs/dynamic-provisioning.yaml",
    )
    parser.add_argument(
        "--profiling-data",
        "-p",
        dest="profiling_data",
        help="Profiling data path.",
        default="AE/data",
    )
    args = parser.parse_args()
    namespace = args.app
    directory = args.profiling_data
    config_path = args.config
    data_path = os.path.join(directory, f"data_{args.app}")
    config = Box.from_yaml(filename=config_path)

    pod_spec = config.pod_spec
    nodes = config.nodes
    parts = config.parts
    cluster_status = pd.DataFrame(config.cluster_status)

    erms_scheduler = ErmsScheduler(data_path)
    pod_spec = {
        "CPU": float(pod_spec["cpu_size"]),
        "mem": parse_mem(pod_spec["mem_size"]),
    }
    cpu_est_df = pd.read_csv(f"{data_path}/fittingResult/cpuUsagePrediction.csv")
    mem_est_df = pd.read_csv(f"{data_path}/fittingResult/memUsagePrediction.csv")
    latency_target_df = pd.read_csv(
        f"{data_path}/scalingResult/ermsPrioLatencyTargetByMS.csv"
    )
    container_df = pd.read_csv(f"{data_path}/scalingResult/ermsPrioContainerByMS.csv")
    data = erms_scheduler.process_data(
        pod_spec, cpu_est_df, mem_est_df, latency_target_df, container_df
    )
    erms_scheduler.strategy(
        data["microservice"].tolist(),
        data["container"].tolist(),
        parts,
        pod_spec,
        cluster_status,
        {"CPU": data["cpu_est"].tolist(), "mem": data["mem_est"].tolist()},
    )
    erms_scheduler.save_result()
    result = erms_scheduler.allocation
    result = result.loc[result["container"] != 0]
    print(result.rename(columns={"container": "number of containers"}))
    print(f"Allocation result is stored in {erms_scheduler.result_path}/allocation.csv")
