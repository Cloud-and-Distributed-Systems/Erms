import ipdb
from time import sleep
import utils.prometheus as prometheus_fetcher
from kubernetes import client, config
import json
import pandas as pd
import re


def read_cluster_data(nodes, prometheus_host):
    config.load_kube_config()
    api = client.CoreV1Api()
    api_resp = api.list_node(_preload_content=False)
    resp_data = json.loads(api_resp.read().decode("utf-8"))["items"]
    # Create a bidirectional mapping dict
    mapping = {}
    for node in resp_data:
        address_1 = node["status"]["addresses"][0]
        address_2 = node["status"]["addresses"][1]
        mapping[address_1["address"]] = address_2["address"]
        mapping[address_2["address"]] = address_1["address"]
    # get capacity data
    resp_df = pd.json_normalize(
        resp_data,
        ["status", "addresses"],
        [["status", "capacity", "cpu"], ["status", "capacity", "memory"]],
    ).rename(
        columns={
            "address": "node",
            "status.capacity.cpu": "cpuCap",
            "status.capacity.memory": "memCap",
        }
    )
    # process capacity data
    capacity_data = resp_df.loc[resp_df["type"] == "Hostname"].drop(columns="type")
    capacity_data["memCap"] = (
        capacity_data["memCap"].str.split("Ki").apply(lambda x: float(x[0]) / 1024)
    )
    # parse node's name to its internal ip address
    ips = [mapping[x] for x in nodes]
    record_path = ["data", "result"]
    # get cpu utilization
    cpu_usage_data = pd.json_normalize(
        prometheus_fetcher.fetch_node_cpu_usage(prometheus_host, ips).json(),
        record_path,
    )[["value", "metric.instance"]]
    cpu_usage_data = {
        "node": cpu_usage_data["metric.instance"]
        .str.split(":")
        .apply(lambda x: mapping[x[0]]),
        "cpuUsage": cpu_usage_data["value"].apply(lambda x: x[1]),
    }
    cpu_usage_data = pd.DataFrame(cpu_usage_data)
    # get memory utilization
    mem_usage_data = pd.json_normalize(
        prometheus_fetcher.fetch_node_mem_usage(prometheus_host, ips).json(),
        record_path,
    )[["value", "metric.instance"]]
    mem_usage_data = {
        "node": mem_usage_data["metric.instance"]
        .str.split(":")
        .apply(lambda x: mapping[x[0]]),
        "memUsage": mem_usage_data["value"].apply(lambda x: x[1]),
    }
    mem_usage_data = pd.DataFrame(mem_usage_data)
    # merge usage data
    usage_data = mem_usage_data.merge(cpu_usage_data, on="node")
    # get cpu allocation
    cpu_aloc = pd.json_normalize(
        prometheus_fetcher.fetch_node_cpu_aloc(prometheus_host, nodes).json(),
        record_path,
    ).rename(columns={"metric.node": "node"})
    cpu_aloc = cpu_aloc.assign(CPU_aloc=cpu_aloc["value"].apply(lambda x: x[1]))
    # get memory allocation
    mem_aloc = pd.json_normalize(
        prometheus_fetcher.fetch_node_mem_aloc(prometheus_host, nodes).json(),
        record_path,
    ).rename(columns={"metric.node": "node"})
    mem_aloc = mem_aloc.assign(mem_aloc=mem_aloc["value"].apply(lambda x: x[1]))
    aloc_data = mem_aloc.merge(cpu_aloc, on="node")
    # calculate allocated cpu and memory
    data = usage_data.merge(capacity_data, on="node").merge(aloc_data, on="node")
    data = {
        "node": data["node"],
        "CPU_used": data["cpuUsage"].astype(float) * data["cpuCap"].astype(float),
        "mem_used": data["memUsage"].astype(float) * data["memCap"].astype(float),
        "CPU_cap": data["cpuCap"].astype(float),
        "mem_cap": data["memCap"].astype(float),
        "CPU_aloc": data["CPU_aloc"].astype(float),
        "mem_aloc": data["mem_aloc"].astype(float),
    }
    return pd.DataFrame(data)


def parse_mem(mem_str):
    mem_str_grp = re.search(r"([0-9\.]*)([^0-9\.]*)", mem_str)
    return (
        float(mem_str_grp.group(1))
        * {"Mi": 1, "Gi": 1024, "Ti": 1024 * 1024}[mem_str_grp.group(2)]
    )


def data_preprocessing(data: pd.DataFrame):
    print(len(data))
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [
    #     ("ComposePost", "media-service"),
    #     ("ComposePost", "user-mention"),
    #     ("ComposePost", "text-service"),
    #     ("ComposePost", "unique-id-service"),
    #     ("ComposePost", "url-shorten-service"),
    #     ("ComposePost", "user-mention-service"),
    #     ("ComposePost", "user-service"),
    #     ("ComposePost", "user-timeline-service"),
    # ]
    # data = data[(~index.isin(indexes)) | (data["reqFreq"] < 120)]

    # Hotel-reservation
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("Recommendation", "frontend")]
    # data = data[(~index.isin(indexes)) | (data["latency"] < 40000)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("Recommendation", "profile")]
    # data = data[(~index.isin(indexes)) | (data["latency"] < 35000)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("Recommendation", "recommendation")]
    # data = data[(~index.isin(indexes)) | (data["latency"] < 50000)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("Search", "profile")]
    # data = data[(~index.isin(indexes)) | (data["latency"] < 20000)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("Search", "geo")]
    # data = data[(~index.isin(indexes)) | (data["latency"] < 300000)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("Search", "search")]
    # data = data[(~index.isin(indexes)) | (data["reqFreq"] < 100)]

    # Media-microsvc
    index = data.set_index(["service", "microservice"]).index
    indexes = [("ComposeReview", "rating-service")]
    data = data[(~index.isin(indexes)) | (data["reqFreq"] < 80)]
    index = data.set_index(["service", "microservice"]).index
    indexes = [("ComposeReview", "unique-id-service")]
    data = data[(~index.isin(indexes)) | (data["reqFreq"] < 100)]
    index = data.set_index(["service", "microservice"]).index
    indexes = [("ComposeReview", "user-review")]
    data = data[(~index.isin(indexes)) | (data["latency"] < 350)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("ComposeReview", "unique-id-service")]
    # data = data[(~index.isin(indexes)) | (data["latency"] < 12000)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("ComposeReview", "user-service")]
    # data = data[(~index.isin(indexes)) | (data["latency"] < 15000)]

    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("HomeTimeline", "post-storage-service")]
    # home_post = data[index.isin(indexes)]
    # data = data[~index.isin(indexes)]
    # index = data.set_index(["service", "microservice"]).index
    # indexes = [("UserTimeline", "post-storage-service")]
    # user_post = data[index.isin(indexes)]
    # data = data[~index.isin(indexes)]
    # home_post = home_post.assign(
    #     targetReqFreq=home_post["targetReqFreq"] / 2, reqFreq=home_post["reqFreq"] / 2
    # )
    # user_post = user_post.assign(
    #     targetReqFreq=user_post["targetReqFreq"] / 3, reqFreq=user_post["reqFreq"] / 3
    # )
    # data = pd.concat([data, home_post, user_post])
    print(len(data))
    ipdb.set_trace()
    return data


def recalculate_contribution(file_path):
    data = pd.read_csv(file_path)
    data = data.assign(original_contribution=data["contribution"])
    no_entrance_data = data.loc[
        (~data["microservice"].str.contains("nginx"))
        & (~data["microservice"].str.contains("frontend"))
    ]
    new_contribution = no_entrance_data.groupby("service").apply(
        lambda x: x["contribution"] / x["contribution"].sum()
    )
    data = (
        data.drop(columns="contribution")
        .merge(
            new_contribution.reset_index()[["level_1", "contribution"]],
            how="left",
            left_index=True,
            right_on="level_1",
        )
        .drop(columns=["level_1"])
        .fillna(-1)
    )
    ipdb.set_trace()


def wait_deployment(namespace, timeout):
    config.load_kube_config()
    api = client.CoreV1Api()
    used_time = 0
    deployment_finished_flag = False
    print("Waiting for deployment finished...")
    while used_time < timeout and not deployment_finished_flag:
        if used_time % 60 == 0 and used_time != 0:
            print(f"{used_time} seconds passed")
        api_resp = api.list_namespaced_pod(namespace, _preload_content=False)
        resp_data = json.loads(api_resp.read().decode("utf-8"))["items"]
        deployment_finished_flag = True
        unfinished_pods = []
        for pod in resp_data:
            pod_finished_flag = True
            if pod["status"]["phase"] != "Running":
                deployment_finished_flag = False
                pod_finished_flag = False
            else:
                for container in pod["status"]["containerStatuses"]:
                    if not container["ready"]:
                        deployment_finished_flag = False
                        pod_finished_flag = False
            if not pod_finished_flag:
                unfinished_pods.append(pod["metadata"]["name"])
        print(f"Unfinished Pods: {', '.join(unfinished_pods)}")
        used_time += 5
        sleep(5)
    if not deployment_finished_flag:
        print("WARNING: Deployment waiting timeout!")
        # ipdb.set_trace()
    else:
        print(f"Deployment finished! Used time: {used_time}s")


def wait_deletion(namespace, timeout):
    config.load_kube_config()
    api = client.CoreV1Api()
    used_time = 0
    deletion_finished_flag = False
    sleep(5)
    print("Waiting for deletion finished...")
    while used_time < timeout and not deletion_finished_flag:
        if used_time % 60 == 0 and used_time != 0:
            print(f"{used_time} seconds passed")
        api_resp = api.list_namespaced_pod(namespace, _preload_content=False)
        resp_data = json.loads(api_resp.read().decode("utf-8"))["items"]
        status_list = [
            x["metadata"]["name"]
            for x in resp_data
            if "deletionTimestamp" in x["metadata"]
        ]
        deletion_finished_flag = True if len(status_list) == 0 else False
        used_time += 5
        sleep(5)
    if not deletion_finished_flag:
        print("WARNING: Deletion waiting timeout!")
        # ipdb.set_trace()
    else:
        print(f"Deletion finished! Used time: {used_time}s")


if __name__ == "__main__":
    usage = input("usage:")
    if usage == "preprocessing":
        path = input("path:")
        data_preprocessing(pd.read_csv(path))
    elif usage == "contribution":
        path = input("path: ")
        recalculate_contribution(path)
    elif usage == "calc-latency":
        pd.read_csv("temp.csv").groupby(["service", "targetThroughput"]).mean().astype(
            int
        ).reset_index()[["service", "targetThroughput", "traceDuration"]].to_csv(
            "temp.csv", index=False
        )
