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
    if "services" in config:
        services = config.services
    else:
        services = None

    merge = ErmsBased(data_path)
    merge.construct_graph(services)
    models = merge.compute_model(cpu_inter, mem_inter)
    merge.update_sharing_ms_models(models)
    for service, span in merge.root_spans.items():
        span.assign_model(models[service])
        span.merge_model()
        span.compute_latency_target(slas[service] * 1000)
        span.merge_model()
        span.compute_latency_target(slas[service] * 1000)
        print(service + "'s Latency Target Computation Result:")
        print(span.to_str(["latency_target"]))
