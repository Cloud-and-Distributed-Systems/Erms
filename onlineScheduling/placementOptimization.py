import math, yaml, os, pickle
from random import random
from time import sleep
from typing import Dict, List
import cvxpy as cp, numpy as np, pandas as pd
from kubernetes import client, config, watch
from utils.others import parse_mem, wait_deployment
from onlineScheduling.applyPriority import apply_priority
import utils.deploymentEditor as editor
from deployment.deployer import delete_by_yaml, deploy_by_yaml
from utils.others import read_cluster_data


class ErmsScheduler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.result_path = f"{data_path}/onlineSchedulingResult"
        os.system(f"rm -rf {self.result_path}")
        os.system(f"mkdir -p {self.result_path}")
        config.load_kube_config()

    def _fake_data(self):
        ms_num = np.random.randint(10, 20)
        ms_list = [f"ms_{x}" for x in range(ms_num)]
        ms_list = ["ms_1", "ms_2", "ms_3", "ms_4", "ms_5"]
        mem_estimation = [100, 150, 80, 180, 100]
        CPU_estimation = [0.05, 0.08, 0.1, 0.05, 0.1]
        container_num_list = [np.random.randint(2, 10) for _ in range(ms_num)]
        container_num_list = [5, 3, 2, 1, 1]
        pod_specific = {"CPU": 0.1, "mem": 200}
        part_num = np.random.randint(5, 10)
        part_num = 1
        cluster_status = pd.DataFrame(
            [
                {
                    "node": "node_1",
                    "CPU_aloc": 1,
                    "mem_aloc": 1024,
                    "CPU_cap": 4,
                    "mem_cap": 2048,
                },
                {
                    "node": "node_2",
                    "CPU_aloc": 2,
                    "mem_aloc": 512,
                    "CPU_cap": 4,
                    "mem_cap": 4096,
                },
                {
                    "node": "node_3",
                    "CPU_aloc": 10,
                    "mem_aloc": 4096,
                    "CPU_cap": 16,
                    "mem_cap": 8192,
                },
            ]
        )
        return {
            "ms_list": ms_list,
            "container_num_list": container_num_list,
            "pod_specific": pod_specific,
            "part_num": part_num,
            "cluster_status": cluster_status,
            "usage_estimation": {"CPU": CPU_estimation, "mem": mem_estimation},
        }

    def main(
        self,
        pod_spec,
        parts,
        nodes,
        yaml_repo,
        namespace,
        prometheus_host,
        app_img,
        cpu_est_df=None,
        mem_est_df=None,
        latency_target_df=None,
        container_df=None,
    ):
        processed_pod_spec = {
            "CPU": float(pod_spec["cpu_size"]),
            "mem": parse_mem(pod_spec["mem_size"]),
        }
        if cpu_est_df is None:
            cpu_est_df = pd.read_csv(
                f"{self.data_path}/fittingResult/cpuUsagePrediction.csv"
            )
        if mem_est_df is None:
            mem_est_df = pd.read_csv(
                f"{self.data_path}/fittingResult/memUsagePrediction.csv"
            )
        if latency_target_df is None:
            latency_target_df = pd.read_csv(
                f"{self.data_path}/scalingResult/ermsPrioLatencyTargetByMS.csv"
            )
        if container_df is None:
            container_df = pd.read_csv(
                f"{self.data_path}/scalingResult/ermsPrioContainerByMS.csv"
            )

        data = self.process_data(
            processed_pod_spec, cpu_est_df, mem_est_df, latency_target_df, container_df
        )
        yamls = editor.read_all_yaml(f"{yaml_repo}/test")
        yamls = editor.base_yaml_preparation(yamls, namespace, app_img, pod_spec)
        editor.save_all_yaml("tmp/scheduledAPP", yamls)
        delete_by_yaml("tmp/scheduledAPP", True, namespace)
        # Waiting for prometheus get the newest data
        sleep(30)
        cluster_status = read_cluster_data(nodes, prometheus_host)
        while len(cluster_status) != len(nodes):
            cluster_status = read_cluster_data(nodes, prometheus_host)
        # Avoid decimal to integer problem
        cluster_status = cluster_status.assign(
            CPU_cap=cluster_status["CPU_cap"] * 0.9,
            mem_cap=cluster_status["mem_cap"] * 0.9,
        )
        # Do not put entrance microservices on test nodes,
        # They will occupy lots of resources and cause lack
        # of resources/unstable of nodes. Put them on infra nodes.
        entrance = data.loc[
            data["microservice"].isin(["nginx-web-server", "frontend", "nginx"])
        ]
        data = data.loc[
            ~data["microservice"].isin(["nginx-web-server", "frontend", "nginx"])
        ]
        self.strategy(
            data["microservice"].tolist(),
            data["container"].tolist(),
            parts,
            processed_pod_spec,
            cluster_status,
            {"CPU": data["cpu_est"].tolist(), "mem": data["mem_est"].tolist()},
        )
        self.prepare_yaml(
            data, yaml_repo, namespace, pod_spec, app_img, nodes, entrance
        )
        deploy_by_yaml("tmp/scheduledAPP")
        self.erms_scheduler(namespace)
        self.wait_until_done(namespace)
        apply_priority(namespace, self.data_path, latency_target_df)

    def process_data(
        self,
        pod_spec,
        cpu_est_df: pd.DataFrame,
        mem_est_df: pd.DataFrame,
        latency_target_df: pd.DataFrame,
        container_df: pd.DataFrame,
    ):
        workload = latency_target_df.groupby("service")["workload"].min().reset_index()
        data = (
            latency_target_df[["service", "microservice"]]
            .drop_duplicates()
            .merge(container_df, on="microservice")
            .merge(workload, on="service")
        )
        data = data.assign(avg_workload=data["workload"] / data["container"])
        cpu_models = {}
        for service, service_df in cpu_est_df.groupby("service"):
            cpu_models[service] = {}
            for _, model_record in service_df.iterrows():
                with open(model_record["modelFile"], "rb") as file:
                    model = pickle.load(file)
                cpu_models[service][model_record["microservice"]] = model
        mem_models = {}
        for service, service_df in mem_est_df.groupby("service"):
            mem_models[service] = {}
            for _, model_record in service_df.iterrows():
                with open(model_record["modelFile"], "rb") as file:
                    model = pickle.load(file)
                mem_models[service][model_record["microservice"]] = model

        cpu_est = []
        mem_est = []
        for _, row in data.iterrows():
            if row["microservice"] in cpu_models[row["service"]]:
                cpu_est.append(
                    cpu_models[row["service"]][row["microservice"]].predict(
                        row["avg_workload"]
                    )
                )
            else:
                cpu_est.append(100)
            if row["microservice"] in mem_models[row["service"]]:
                mem_est.append(
                    mem_models[row["service"]][row["microservice"]].predict(
                        row["avg_workload"]
                    )
                )
            else:
                mem_est.append(100)
        data = data.assign(cpu_est=cpu_est, mem_est=mem_est)
        data = (
            data.groupby("microservice")[["cpu_est", "mem_est", "container"]]
            .max()
            .reset_index()
        )
        data["cpu_est"] = data["cpu_est"] * pod_spec["CPU"] / 100
        data["mem_est"] = data["mem_est"] * pod_spec["mem"] / 100
        return data

    def strategy(
        self,
        ms_list,
        container_num_list,
        part_num,
        pod_specific,
        cluster_status: pd.DataFrame,
        usage_estimation,
    ):
        CPU_estimation = usage_estimation["CPU"]
        mem_estimation = usage_estimation["mem"]
        node_list = cluster_status["node"].tolist()
        node_num = len(node_list)
        ms_num = len(ms_list)

        # Divide all nodes into different small parts to speed up
        # calculation for large problem.
        # "p_" means partitioned
        nodes_per_part = math.floor(node_num / part_num)
        if nodes_per_part <= 0:
            raise Exception("Too many parts, too less nodes!")
        p_node_list = []
        for i in range(part_num - 1):
            p_node_list.append(
                [node_list[i * nodes_per_part + x] for x in range(nodes_per_part)]
            )
        rest = node_num - len(p_node_list) * nodes_per_part
        p_node_list.append(node_list[-rest:])

        p_container_list = [x / part_num for x in container_num_list]

        allocation_result = pd.DataFrame(columns=["microservice", "node", "container"])
        for _, p_nodes in enumerate(p_node_list):
            # some parts may not fully filled, so need to know the exact
            # length of nodes instead of using nodes_per_part
            p_node_num = len(p_nodes)

            # Allocated cpu resources, i.e. sum(pod_limits) by(node)
            p_CPU_aloc = cluster_status.loc[cluster_status["node"].isin(p_nodes)][
                "CPU_aloc"
            ].tolist()
            # Real used cpu resources, i.e. node_capacity * node_usage
            p_CPU_used = cluster_status.loc[cluster_status["node"].isin(p_nodes)][
                "CPU_used"
            ].tolist()
            # Node cpu allocation capacity
            p_CPU_cap = cluster_status.loc[cluster_status["node"].isin(p_nodes)][
                "CPU_cap"
            ].tolist()
            # Allocated memory resources, i.e. sum(pod_limits) by(node)
            p_mem_aloc = cluster_status.loc[cluster_status["node"].isin(p_nodes)][
                "mem_aloc"
            ].tolist()
            # Real used memory resources, i.e. node_capacity * node_usage
            p_mem_used = cluster_status.loc[cluster_status["node"].isin(p_nodes)][
                "mem_used"
            ].tolist()
            # Node memory allocation capacity
            p_mem_cap = cluster_status.loc[cluster_status["node"].isin(p_nodes)][
                "mem_cap"
            ].tolist()

            # Each matrix's row's length equals the number of nodes
            # multiply the number of microservices. The first matrix
            # is used to represents that all microservices' needed
            # containers are scheduled. The second matrix is used to
            # represents that scheduling result doesn't exceed the
            # capacity of each node.
            product = p_node_num * ms_num
            """
            matrix_1, e.g.
            [1, 0, 0, 1, 0, 0, 1, 0, 0]
            [0, 1, 0, 0, 1, 0, 0, 1, 0]
            [0, 0, 1, 0, 0, 1, 0, 0, 1]
            matrix_2_*, e.g.
            [1, 1, 1, 0, 0, 0, 0, 0, 0]
            [0, 0, 0, 1, 1, 1, 0, 0, 0]
            [0, 0, 0, 0, 0, 0, 1, 1, 1]
            """
            matrix_1 = np.zeros([ms_num, product])
            matrix_2_CPU = np.zeros([p_node_num, product])
            matrix_2_mem = np.zeros([p_node_num, product])
            est_CPU_matrix = np.zeros([p_node_num, product])
            est_mem_matrix = np.zeros([p_node_num, product])
            allocation = cp.Variable(product)
            for i in range(ms_num):
                matrix_1[i][[x * ms_num + i for x in range(p_node_num)]] = 1
            for i in range(p_node_num):
                matrix_2_CPU[i][
                    [x for x in range(ms_num * i, ms_num * (i + 1))]
                ] = pod_specific["CPU"]
                est_CPU_matrix[i][
                    [x for x in range(ms_num * i, ms_num * (i + 1))]
                ] = CPU_estimation
                matrix_2_mem[i][
                    [x for x in range(ms_num * i, ms_num * (i + 1))]
                ] = pod_specific["mem"]
                est_mem_matrix[i][
                    [x for x in range(ms_num * i, ms_num * (i + 1))]
                ] = mem_estimation

            est_p_CPU_used = est_CPU_matrix @ allocation + p_CPU_used
            est_p_mem_used = est_mem_matrix @ allocation + p_mem_used
            est_p_cluster_CPU_usage = sum(est_p_CPU_used) / sum(p_CPU_cap)
            est_p_cluster_mem_usage = sum(est_p_mem_used) / sum(p_mem_cap)
            # The objective is to minimize the gap between
            # each node's resource usage and the cluster(part)
            # wide average resource usage
            CPU_obj = cp.sum_squares(
                est_p_CPU_used / p_CPU_cap - est_p_cluster_CPU_usage
            )
            mem_obj = cp.sum_squares(
                est_p_mem_used / p_mem_cap - est_p_cluster_mem_usage
            )
            obj = cp.Minimize(CPU_obj + mem_obj)

            """
            constraint_1: All pods need to be allocated
            constraint_2: After allocation, node's CPU doesn't exceed its limitation
            constraint_3: After allocation, node's mem doesn't exceed its limitation
            constraint_4: The allocation result cannot be negative
            """
            constraint_1 = matrix_1 @ allocation == p_container_list
            constraint_2 = matrix_2_CPU @ allocation + p_CPU_aloc <= p_CPU_cap
            constraint_3 = matrix_2_mem @ allocation + p_mem_aloc <= p_mem_cap
            constraint_4 = allocation >= 0
            constraints = [constraint_1, constraint_2, constraint_3, constraint_4]
            problem = cp.Problem(obj, constraints)
            problem.solve()
            if allocation.value is None:
                raise BaseException("Allocation failed! Cluster resources not enough")
            for ms_index, ms in enumerate(ms_list):
                ms_allocation = []
                for node_index, node in enumerate(p_nodes):
                    ms_allocation.append(
                        {
                            "microservice": ms,
                            "node": node,
                            "container": allocation.value[
                                node_index * ms_num + ms_index
                            ],
                        }
                    )
                allocation_result = pd.concat(
                    [allocation_result, pd.DataFrame(ms_allocation)]
                )
        allocation_result = allocation_result.reset_index().drop(columns="index")

        def decimal_to_int(decimal_container_data: pd.DataFrame):
            int_result = pd.DataFrame(
                {
                    "node": decimal_container_data["node"],
                }
            )
            int_container = decimal_container_data["container"].astype(int).tolist()
            decimal: pd.Series = decimal_container_data[
                "container"
            ] - decimal_container_data["container"].astype(int)
            unallocated = round(decimal.sum())
            decimal = decimal / unallocated
            cumulative = 0

            def calculate_pdf(prob):
                nonlocal cumulative
                new_prob = cumulative + prob
                cumulative += prob
                return new_prob

            pdf = decimal.apply(calculate_pdf).reset_index()["container"]
            for _ in range(unallocated):
                random_value = random()
                int_container[pdf.loc[pdf >= random_value].index[0]] += 1
            int_result = int_result.assign(container=int_container)
            return int_result

        self.allocation = (
            allocation_result.groupby("microservice")
            .apply(decimal_to_int)
            .reset_index()[["microservice", "node", "container"]]
        )

    def erms_scheduler(self, namespace):
        api = client.CoreV1Api()
        watcher = watch.Watch()

        def containers_to_node_list(microservice_group: pd.DataFrame):
            node_list = []
            microservice_group.apply(
                lambda x: node_list.extend([x["node"]] * x["container"]),
                axis=1,
            )
            microservice = microservice_group["microservice"].unique().item()
            possible_nodes[microservice] = node_list
            return None

        possible_nodes: Dict(List(str)) = {}
        self.allocation.groupby("microservice").apply(containers_to_node_list)
        if "nginx-web-server" in possible_nodes:
            possible_nodes["nginx-thrift"] = possible_nodes["nginx-web-server"]
        if "nginx" in possible_nodes:
            possible_nodes["nginx-web-server"] = possible_nodes["nginx"]

        ready_pods = []
        for event in watcher.stream(api.list_namespaced_pod, namespace):
            if (
                event["object"].status.phase == "Pending"
                and event["object"].spec.scheduler_name == "erms-scheduler"
                and event["object"].metadata.name not in ready_pods
            ):
                pod_name = str(event["object"].metadata.name)
                pod_owner = "-".join(pod_name.split("-")[:-2])
                try:
                    target = client.V1ObjectReference()
                    target.kind = "Node"
                    target.api_version = "v1"
                    target.name = possible_nodes[pod_owner].pop()
                    target.namespace = namespace
                    body = client.V1Binding(target=target)
                except:
                    import ipdb

                    ipdb.set_trace()
                    watcher.stop()
                    print(f"Failed to allocate:{pod_name}")
                    continue

                meta = client.V1ObjectMeta()
                meta.name = pod_name

                body.target = target
                body.metadata = meta
                api.create_namespaced_binding(namespace, body, _preload_content=False)
                ready_pods.append(pod_name)

                rest_node = sum([len(possible_nodes[x]) for x in possible_nodes])
                if rest_node == 0:
                    watcher.stop()
                    return True

    def save_result(self):
        self.allocation.to_csv(f"{self.result_path}/allocation.csv")

    def prepare_yaml(
        self,
        scaling_data: pd.DataFrame,
        yaml_repo,
        namespace,
        pod_spec,
        app_img,
        nodes,
        entrance,
    ):
        # Only those ms that have containers assigned are set to use erms-scheduler
        yaml_list = editor.read_all_yaml(f"{yaml_repo}/test")
        yaml_list = editor.base_yaml_preparation(
            yaml_list, namespace, app_img, pod_spec
        )
        yaml_list = editor.assign_containers(yaml_list, scaling_data)
        yaml_list = editor.assign_containers(yaml_list, entrance)
        yaml_list = editor.assign_affinity(yaml_list, nodes)
        with open("yamlRepository/templates/deploymentAffinity.yaml", "r") as file:
            node_affinity = yaml.load(
                file.read().replace(
                    "%%%", "[izj6chnuk65flw16wpdx1wz, izj6c4ghaaduar69lb5hlsz]"
                ),
                yaml.CLoader,
            )
        path = "spec.template.spec.affinity"
        value = {
            "nginx-thrift": node_affinity,
            "frontend": node_affinity,
            "nginx": node_affinity,
        }
        yaml_list = editor.insert_to_python_objs(
            path, value, yaml_list, key_path="metadata.name"
        )
        path = "spec.template.spec.schedulerName"
        value = dict(
            zip(scaling_data["microservice"], ["erms-scheduler"] * len(scaling_data))
        )
        if "nginx-web-server" in value:
            value["nginx-thrift"] = value["nginx-web-server"]
        if "nginx" in value:
            value["nginx-web-server"] = value["nginx"]
        key_path = "metadata.name"
        yaml_list = editor.insert_to_python_objs(
            path, value, yaml_list, key_path=key_path
        )

        os.system("rm -rf tmp/scheduledAPP")
        os.system("mkdir -p tmp/scheduledAPP")
        editor.save_all_yaml("tmp/scheduledAPP", yaml_list)

    def wait_until_done(self, namespace):
        wait_deployment(namespace, 300)
