import os

from box import Box
from onlineScaling.erms import ErmsBased
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
        default="AE/scripts/configs/priority-scheduling.yaml",
    )
    parser.add_argument(
        "--profiling-data",
        "-p",
        dest="profiling_data",
        help="Profiling data path.",
        default="AE/data",
    )
    args = parser.parse_args()
    directory = args.profiling_data
    config_path = args.config
    data_path = os.path.join(directory, f"data_{args.app}")
    config = Box.from_yaml(filename=config_path)

    cpu_inter = config.interferences.cpu
    mem_inter = config.interferences.mem
    slas = config.slas
    workloads = config.workloads

    workloads_dict = {key: {"default": value} for key, value in workloads.items()}
    slas = {x[0]: x[1] * 1000 for x in slas.items()}
    if "services" in config:
        services = config.services
    else:
        services = None

    merge = ErmsBased(data_path)
    merge.construct_graph(services)
    db_data = merge.process_db_data()
    models = merge.compute_model(cpu_inter, mem_inter)
    merge.init_span_data(workloads_dict, models, db_data)
    merge.update_sharing_ms_models(models)
    merge.calculate_latency_targets_and_containers(slas)
    if len(merge.root_spans.keys()) >= 1:
        merge.priority_scheduling(workloads_dict, slas)
    for service, span in merge.root_spans.items():
        print(service + "'s Priority Scheduling Result:")
        print(span.to_str(["latency_target", "container"]))
    merge.save_to_csv("ermsPrioLatencyTargetByMS.csv", "ermsPrioContainerByMS.csv")
    print(
        f"Latency target data is stored in {data_path}/scalingResult/ermsPrioLatencyTargetByMS.csv"
    )
    print(
        f"Container data is stored in {data_path}/scalingResult/ermsPrioContainerByMS.csv"
    )
