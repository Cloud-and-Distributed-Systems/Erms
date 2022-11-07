import configs
from testing.testCollection import init_app, full_init
from offlineProfiling.fitting import Fitting
from onlineScaling.erms import ErmsBased
from onlineScheduling.placementOptimization import ErmsScheduler


def online_scaling(
    cpu_inter=configs.SCALING_CONFIG.cpu_interference,
    mem_inter=configs.SCALING_CONFIG.mem_interference,
    services=configs.SCALING_CONFIG.services,
    slas=configs.SCALING_CONFIG.slas,
    data_path=configs.GLOBAL_CONFIG.data_path,
    workloads=configs.SCALING_CONFIG.workloads.items(),
):
    workloads_dict = {key: {"default": value} for key, value in workloads}

    merge = ErmsBased(data_path)
    merge.non_priority_main(cpu_inter, mem_inter, workloads_dict, slas, services)
    merge.priority_main(cpu_inter, mem_inter, workloads_dict, slas, services)


def offline_fitting():
    range_dict = {}
    for service, cutoff_range in dict(configs.FITTING_CONFIG.cutoff_range).items():
        range_dict[service] = range(
            cutoff_range.min, cutoff_range.max, cutoff_range.step
        )
    fitter = Fitting(
        configs.GLOBAL_CONFIG.data_path,
        configs.FITTING_CONFIG.percentage_for_train,
        range_dict,
        configs.GLOBAL_CONFIG.figure_path,
        configs.FITTING_CONFIG.cpu_limits,
        configs.FITTING_CONFIG.min_of_max_cpu,
        configs.FITTING_CONFIG.acceptable_throughtput_error_rate,
        configs.FITTING_CONFIG.throughput_classification_precision,
    )
    fitter.erms_main()


def online_scheduling():
    erms_scheduler = ErmsScheduler(configs.GLOBAL_CONFIG.data_path)
    erms_scheduler.main(
        configs.GLOBAL_CONFIG.pod_spec,
        configs.SCALING_CONFIG.partitions,
        configs.GLOBAL_CONFIG.nodes_for_test,
        configs.GLOBAL_CONFIG.yaml_repo_path,
        configs.GLOBAL_CONFIG.namespace,
    )

# ==============================================================================================
# Fully initialize application, redeploy all stateful & stateless deployments
# The first parameter is application name, possible choices: ["social", "media", "hotel"]
# The second parameter is the entrance port of that application, default value is shown below

# Before processing this step, please check the configs/*-global.yaml file to check your
# configuration settings are correct, and do not forget to set environment variable ERMS_APP

# full_init("social", 30628)
# full_init("media", 30092)
# full_init("hotel", 30096)
# ==============================================================================================
# Initialize application, redeploy all stateless deployments only

# init_app()
# ==============================================================================================