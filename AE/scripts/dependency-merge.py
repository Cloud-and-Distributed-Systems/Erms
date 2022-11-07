import os
from onlineScaling.erms import ErmsBased
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online scaling, dependency merge part",
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
        "--service",
        "-s",
        dest="service",
        help=(
            "Service name. Different applications have different available services:\n"
            "\thotel-reserv: `Recommendation`, `Search`\n"
            "\tsocial-network: `UserTimeline`, `ComposePost`, `HomeTimeline`\n"
            "\tmedia-microsvc: `ComposeReview`\n"
            "\tFitting all services: `All`\n"
        ),
        default="All",
    )
    parser.add_argument(
        "--profiling-data",
        "-p",
        dest="profiling_data",
        help="Profiling data path.",
        default="AE/data",
    )
    parser.add_argument(
        "--cpu",
        "-c",
        dest="cpu",
        help="Number of CPU interferences.",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--mem",
        "-m",
        dest="mem",
        help="Number of memory interferences.",
        default=6,
        type=int,
    )
    args = parser.parse_args()
    service = args.service
    directory = args.profiling_data
    data_path = os.path.join(directory, f"data_{args.app}")
    cpu_inter = args.cpu
    mem_inter = args.mem

    if service == "All":
        service = None
    else:
        service = [service]

    merge = ErmsBased(data_path)
    merge.construct_graph(service)
    models = merge.compute_model(cpu_inter * 0.4, mem_inter * 800)
    for service, span in merge.root_spans.items():
        span.assign_model(models[service])
        print(service + "'s Original Graph: ")
        print(span.to_str(["model"]))
        span.merge_model()
        print(service + "'s Merged Graph:")
        print(span.to_str(["merged_model"]))
