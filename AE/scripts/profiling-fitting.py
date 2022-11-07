import argparse
import os
import configs
from offlineProfiling.fitting import Fitting
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline Profiling, fitting part",
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
            "\tmedia-microsvc: `ComposeReview`"
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
        "--dir",
        "-d",
        dest="directory",
        help="Path to save profiling data.",
        default="AE/data",
    )
    args = parser.parse_args()
    service = args.service
    input_path = args.profiling_data
    output_path = f"{args.directory}/data_{args.app}"
    data_path = f"{input_path}/data_{args.app}"
    figure_path = f"{data_path}/figures"
    cpu_limits = {
        "hotel-reserv": {"lower": 0.2, "higher": 0.9},
        "social-network": {"lower": 0.2, "higher": 0.9},
        "media-microsvc": {"lower": 0.2, "higher": 0.9},
    }[args.app]

    os.system(f"mv {data_path}/fittingResult {data_path}/fittingResult_original")
    os.system(f"mkdir -p {output_path}")

    fitter = Fitting(
        data_path,
        configs.FITTING_CONFIG.percentage_for_train,
        {
            x[0]: range(x[1].min, x[1].max, x[1].step)
            for x in configs.FITTING_CONFIG.cutoff_range.items()
        },
        figure_path,
        cpu_limits,
        configs.FITTING_CONFIG.min_of_max_cpu,
        configs.FITTING_CONFIG.acceptable_throughtput_error_rate,
        configs.FITTING_CONFIG.throughput_classification_precision,
        configs.GLOBAL_CONFIG.replicas,
        target_services=service,
    )
    fitter.erms_main()

    os.system(f"mv {data_path}/fittingResult {output_path}")
    os.system(f"mv {data_path}/fittingResult_original {data_path}/fittingResult")

    print("Fitting Success!")
    print(f"Fitting results can be found in {output_path}/fittingResult/ermsFull.csv")
