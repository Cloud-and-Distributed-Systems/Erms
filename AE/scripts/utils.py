import multiprocessing
from time import time
from infGenerator.idleInf import IdleInf
from infGenerator.busyInf import BusyInf
from onlineScaling.erms import ErmsBased
from typing import Dict, List, Tuple, Union
from deployment.deployer import delete_by_yaml, deploy_by_yaml
import utils.deploymentEditor as editor
import pandas as pd
import ipdb
from utils.others import parse_mem, wait_deployment

from workloadGenerator.staticWorkload import StaticWorkloadGenerator
import yaml


def container_computation(
    data_path: str,
    slas_workloads: List[Tuple[Dict[str, float], Dict[str, float]]],
    cpu_inter: float,
    mem_inter: float,
    services: List[str],
    methods: List[str],
    prio: Union[bool, None] = None,
    init_containers: Union[Dict[str, int], None] = None,
    base_workload: Union[Dict[str, int], None] = None,
    fixed_ms: Union[Dict[str, int], None] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate containers for different workloads and SLAs
    under certain CPU and memory interference with certain methods.

    Args:
        data_path (str): Params to initialize latency target computation object.
        slas_workloads (List[Tuple[Dict[str, float], Dict[str, float]]]):
        List of tuples, the first element of tuple is SLA,
        and the second one is workload, SLA's units: ms.
        cpu_inter (float): CPU interference, units: core.
        mem_inter (float): Memory interference, units: MiB.
        services (List[str]): Searvices that need computation.
        methods (List[str]): How to compute latency target, possible values: `erms`
        prio (Union[bool, None], optional): Using priority scheduling or not. None means only use priority to erms.
        init_containers (Union[Dict[str, int], None], optional): Firm only, used to initialize microservice containers.
        base_workload(Union[Dict[str, int], None], optional): Firm only, the max workload for init_containers.
        fixed_ms (Union[Dict[str, int], None], optional): Firm only, used to specify microservices that will assign a fixed containers.
        debug (bool, optional): Debug mode indicator. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Latency target result and container result
    """
    container_result = []
    latency_target_result = []
    for var_index, (sla, workload) in enumerate(slas_workloads):
        sla_in_us = {}
        workload_dict = {}
        extra_cols = {
            "cpu_inter": cpu_inter,
            "mem_inter": mem_inter,
            "var_index": var_index,
        }
        for service in services:
            sla_in_us[service] = sla[service] * 1000
            workload_dict[service] = {"default": workload[service]}
            # Add extra information to result
            extra_cols.update(
                {
                    f"{service}_workload": workload[service],
                    f"{service}_sla": sla[service],
                }
            )
        method_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for method in methods:
            calculator = ErmsBased(data_path)
            if prio is None or prio:
                calculator.priority_main(
                    cpu_inter, mem_inter, workload_dict, sla_in_us, services
                )
            else:
                calculator.non_priority_main(
                    cpu_inter, mem_inter, workload_dict, sla_in_us, services
                )
            latency_targets, containers = calculator.generate_result()
            containers.at[
                containers.loc[
                    containers["microservice"].isin(
                        ["frontend", "nginx", "nginx-web-server"]
                    )
                ].index.tolist(),
                "container",
            ] = 20
            if debug:
                ipdb.set_trace()
            extra_cols.update({"method": method})
            latency_target_result.append(latency_targets.assign(**extra_cols))
            container_result.append(containers.assign(**extra_cols))
        # Limits rhythm/grandSLAm/firm's containers
    latency_target_result = pd.concat(latency_target_result)
    container_result = pd.concat(container_result)
    if debug:
        ipdb.set_trace()
    return (latency_target_result, container_result)


def k8s_scheduling(
    containers,
    nodes,
    namespace,
    pod_spec,
    yaml_repo,
    image,
    tmp_folder="tmp/scheduledAPP",
):
    yaml_list = editor.read_all_yaml(f"{yaml_repo}/test")

    yaml_list = editor.base_yaml_preparation(yaml_list, namespace, image, pod_spec)
    yaml_list = editor.assign_containers(yaml_list, containers)
    yaml_list = editor.assign_affinity(yaml_list, nodes)
    with open("yamlRepository/templates/deploymentAffinity.yaml", "r") as file:
        node_affinity = yaml.load(
            file.read().replace(
                "%%%", "[izj6chnuk65flw16wpdx1wz, izj6c4ghaaduar69lb5hlsz]"
            ),
            yaml.CLoader,
        )
    path = "spec.template.spec.affinity"
    value = {"nginx-thrift": node_affinity, "frontend": node_affinity}
    yaml_list = editor.insert_to_python_objs(
        path, value, yaml_list, key_path="metadata.name"
    )
    editor.save_all_yaml(tmp_folder, yaml_list)

    delete_by_yaml(tmp_folder)
    deploy_by_yaml(tmp_folder, True, namespace)


def generate_multiple_workloads(
    target_services, test_duration, workload_configs, scripts, host
):
    generator_processes = []
    for service in target_services:
        workload_generator = StaticWorkloadGenerator(
            workload_configs[service]["thread"],
            workload_configs[service]["conn"],
            test_duration,
            workload_configs[service]["throughput"],
            "wrk2/wrk",
            scripts[service],
            host,
        )
        process = multiprocessing.Process(
            target=workload_generator.generateWorkload,
            args=(f"validation_{service}", workload_configs[service]["clients"]),
        )
        process.start()
        generator_processes.append(process)
    start_time = time()
    for p in generator_processes:
        p.join()
    return start_time


def deploy_infs(interference, nodes):
    for inf_type in ["cpu", "mem"]:
        idle_inf = IdleInf(
            interference[inf_type]["cpu_size"],
            interference[inf_type]["mem_size"],
            inf_type,
        )
        replicas = {
            x[0]: x[1]["idle"] for x in interference[inf_type]["allocation"].items()
        }
        idle_inf.delete_infs(nodes)
        idle_inf.deploy_infs(nodes, replicas)

    for node in nodes:
        args = {"cpu": ["360000s"], "mem": ["360000s", "wired", "360000s"]}
        for inf_type in ["cpu", "mem"]:
            busy_inf = BusyInf(
                [node],
                interference[inf_type]["cpu_size"],
                interference[inf_type]["mem_size"],
                inf_type.replace("mem", "memory"),
                args[inf_type],
            )
            busy_inf.clearAllInterference()
            busy_inf.generateInterference(
                interference[inf_type]["allocation"][node]["busy"]
            )
    wait_deployment("interference", 300)


def get_inter(interference):
    cpu_inter = (
        sum([x[1]["busy"] for x in interference["cpu"]["allocation"].items()])
        * interference["cpu"]["cpu_size"]
        / len(interference["cpu"]["allocation"])
    )
    mem_inter = (
        sum([x[1]["busy"] for x in interference["mem"]["allocation"].items()])
        * parse_mem(interference["mem"]["mem_size"])
        / len(interference["mem"]["allocation"])
    )
    return cpu_inter, mem_inter


CONFIG = {
    "yaml_repo": {
        "media-microsvc": "yamlRepository/mediaMicroservice",
        "social-network": "yamlRepository/socialNetwork",
        "hotel-reserv": "yamlRepository/hotelReservation",
    },
    "data_path": {
        "social-network": "AE/data/data_social-network",
        "hotel-reserv": "AE/data/data_hotel-reserv",
        "media-microsvc": "AE/data/data_media-microsvc",
    },
    "scripts": {
        "media-microsvc": {
            "ComposeReview": "wrk2/scripts/media-microservice/compose-review.lua"
        },
        "social-network": {
            "ComposePost": "wrk2/scripts/social-network/compose-post.lua",
            "HomeTimeline": "wrk2/scripts/social-network/read-home-timeline.lua",
            "UserTimeline": "wrk2/scripts/social-network/read-user-timeline.lua",
        },
        "hotel-reserv": {
            "Search": "wrk2/scripts/hotel-reservation/search.lua",
            "Recommendation": "wrk2/scripts/hotel-reservation/recommendation.lua",
            "Login": "wrk2/scripts/hotel-reservation/login.lua",
        },
    },
    "jaeger_host": {
        "media-microsvc": "http://localhost:30093",
        "social-network": "http://localhost:30094",
        "hotel-reserv": "http://localhost:30095",
    },
    "entry_point": {
        "social-network": "nginx-web-server",
        "hotel-reserv": "frontend",
        "media-microsvc": "nginx",
    },
    "host": {
        "media-microsvc": "http://localhost:30092",
        "social-network": "http://localhost:30628",
        "hotel-reserv": "http://localhost:30096",
    },
    "operations": {
        "media-microsvc": {"ComposeReview": "/wrk2-api/review/compose"},
        "social-network": {
            "ComposePost": "/wrk2-api/post/compose",
            "HomeTimeline": "/wrk2-api/home-timeline/read",
            "UserTimeline": "/wrk2-api/user-timeline/read",
        },
        "hotel-reserv": {
            "Search": "HTTP GET /hotels",
            "Login": "HTTP GET /user",
            "Recommendation": "HTTP GET /recommendations",
        },
    },
    "image": {
        "media-microsvc": "nicklin9907/erms:mediamicroservice-1.0",
        "social-network": "nicklin9907/erms:socialnetwork-1.1",
        "hotel-reserv": "nicklin9907/erms:hotelreservation-1.0",
    },
    "nodes": [
        "izj6c6vb9bfm8mxnvb4n47z",
        "izj6c6vb9bfm8mxnvb4n46z",
        "izj6c6vb9bfm8mxnvb4n45z",
        "izj6c6vb9bfm8mxnvb4n44z",
    ],
    "pod_spec": {"cpu_size": 0.1, "mem_size": "200Mi"},
    "namespace": {
        "social-network": "social-network",
        "hotel-reserv": "hotel-reserv",
        "media-microsvc": "media-microsvc",
    },
    "prometheus_host": "http://localhost:30090",
    "interference": [
        {
            "cpu": {
                "cpu_size": 0.5,
                "mem_size": "10Mi",
                "allocation": {
                    "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 5},
                    "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 5},
                    "izj6c6vb9bfm8mxnvb4n45z": {"idle": 6, "busy": 0},
                    "izj6c6vb9bfm8mxnvb4n44z": {"idle": 6, "busy": 0},
                },
            },
            "mem": {
                "cpu_size": 0.01,
                "mem_size": "500Mi",
                "allocation": {
                    "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 5},
                    "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 5},
                    "izj6c6vb9bfm8mxnvb4n45z": {"idle": 6, "busy": 0},
                    "izj6c6vb9bfm8mxnvb4n44z": {"idle": 6, "busy": 0},
                },
            },
        },
        {
            "cpu": {
                "cpu_size": 0.5,
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
        {
            "cpu": {
                "cpu_size": 0.5,
                "mem_size": "10Mi",
                "allocation": {
                    "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 6},
                    "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 6},
                    "izj6c6vb9bfm8mxnvb4n45z": {"idle": 4, "busy": 0},
                    "izj6c6vb9bfm8mxnvb4n44z": {"idle": 4, "busy": 0},
                },
            },
            "mem": {
                "cpu_size": 0.01,
                "mem_size": "500Mi",
                "allocation": {
                    "izj6c6vb9bfm8mxnvb4n47z": {"idle": 0, "busy": 6},
                    "izj6c6vb9bfm8mxnvb4n46z": {"idle": 0, "busy": 6},
                    "izj6c6vb9bfm8mxnvb4n45z": {"idle": 4, "busy": 0},
                    "izj6c6vb9bfm8mxnvb4n44z": {"idle": 4, "busy": 0},
                },
            },
        }
    ],
    "services": {
        "hotel-reserv": ["Search", "Recommendation"],
        "social-network": ["UserTimeline", "HomeTimeline"],
        "media-microsvc": ["ComposeReview"],
    },
    "fixed_ms": {
        "hotel-reserv": {"frontend": 10},
        "social-network": {"nginx-web-server": 30},
        "media-microsvc": {"nginx": 10},
    },
    "workload_config": {"social-network": {"thread": 1, "conn": 2, "throughput": 4}},
    "slas_workload_configs": {
        "hotel-reserv": [
            (
                {"Search": a, "Recommendation": b},
                {
                    "Search": {"clients": c, "thread": 2, "conn": 4, "throughput": 8},
                    "Recommendation": {
                        "clients": d,
                        "thread": 7,
                        "conn": 14,
                        "throughput": 28,
                    },
                },
            )
            for a in [200, 300]
            for b in [80, 90]
            for c in [10, 20]
            for d in [5, 7]
        ],
        "social-network": [
            (
                {"UserTimeline": a, "HomeTimeline": b},
                {
                    "UserTimeline": {
                        "clients": c,
                        "thread": 2,
                        "conn": 4,
                        "throughput": 7,
                    },
                    "HomeTimeline": {
                        "clients": d,
                        "thread": 2,
                        "conn": 3,
                        "throughput": 6,
                    },
                },
            )
            for a in [200, 300]
            for b in [300, 400]
            for c in [20, 30]
            for d in [20, 30]
        ],
        "media-microsvc": [
            (
                {"ComposeReview": 200},
                {
                    "ComposeReview": {
                        "clients": 8,
                        "thread": 2,
                        "conn": 3,
                        "throughput": 6,
                    }
                },
            ),
            (
                {"ComposeReview": 200},
                {
                    "ComposeReview": {
                        "clients": 10,
                        "thread": 2,
                        "conn": 3,
                        "throughput": 6,
                    }
                },
            ),
            (
                {"ComposeReview": 300},
                {
                    "ComposeReview": {
                        "clients": 10,
                        "thread": 2,
                        "conn": 3,
                        "throughput": 6,
                    }
                },
            ),
            (
                {"ComposeReview": 300},
                {
                    "ComposeReview": {
                        "clients": 20,
                        "thread": 2,
                        "conn": 3,
                        "throughput": 6,
                    }
                },
            ),
        ],
    },
    "init_containers": {
        "hotel-reserv": {
            "Search": {"profile": 2},
            "Recommendation": {"profile": 2},
        },
        "social-network": {
            "ComposePost": {"compose-post-service": 4, "text-service": 2},
            "UserTimeline": {"user-timeline-service": 2, "post-storage-service": 6},
            "HomeTimeline": {"post-storage-service": 3},
        },
        "media-microsvc": {
            "ComposeReview": {"compose-review-service": 3, "movie-id-service": 2}
        },
    },
    "base_workload": {
        "hotel-reserv": {"Search": 60, "Recommendation": 140},
        "social-network": {"ComposePost": 60, "UserTimeline": 140, "HomeTimeline": 120},
        "media-microsvc": {"ComposeReview": 120},
    },
}


def config_to_workload(
    slas_workload_configs: List[Tuple[Dict[str, int], Dict[str, int]]]
):
    slas_workloads = []
    for sla, workload_config in slas_workload_configs:
        workload = {}
        for service in workload_config:
            service_workload_config = workload_config[service]
            workload[service] = (
                service_workload_config["clients"]
                * service_workload_config["throughput"]
            )
        slas_workloads.append((sla, workload))
    return slas_workloads
