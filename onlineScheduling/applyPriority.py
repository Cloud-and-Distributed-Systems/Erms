import os
import pandas as pd
from kubernetes import client, config

config.kube_config.load_kube_config()


def apply_priority(namespace, data_path, latency_target_df=None):
    if latency_target_df is None:
        latency_target_df = pd.read_csv(
            f"{data_path}/scalingResult/ermsPrioLatencyTargetByMS.csv"
        )
    priority = calculate_priority(latency_target_df)
    if len(priority) != 0:
        parent_data, target_data = merge_k8s_data_to_priority(namespace, priority)
        rules, virtual_if_count = generate_if_data(priority, parent_data, target_data)
        generate_scripts(virtual_if_count, rules, data_path)
        deploy_vifs(data_path)


def calculate_priority(latency_targets: pd.DataFrame):
    def assign_priority(x: pd.DataFrame):
        x = x.drop_duplicates(subset=["service", "microservice"])
        sorted_df = x.sort_values("latency_target").dropna()
        if len(sorted_df) <= 1:
            return
        return sorted_df.assign(priority=range(len(sorted_df))).drop(
            columns="microservice"
        )

    priority = latency_targets.groupby("microservice").apply(assign_priority)
    if len(priority) == 0:
        return priority
    return priority.reset_index().drop(columns="level_1")[
        ["service", "microservice", "parent", "priority"]
    ]


def merge_k8s_data_to_priority(namespace, priority: pd.DataFrame):
    data_columns = ["microservice", "pod_ip", "host_ip", "container_id", "cpu", "mem"]
    target_data = pd.DataFrame(columns=data_columns)
    parent_data = pd.DataFrame(columns=data_columns)
    target_ms_set = set(priority["microservice"].unique().tolist())
    parent_ms_set = set(priority["parent"].unique().tolist())
    v1 = client.CoreV1Api()
    res = v1.list_namespaced_pod(namespace=namespace, watch=False)
    for i in res.items:
        pod_name = "-".join(str(i.metadata.name).split("-")[:-2])
        pod_ip = i.status.pod_ip
        host_ip = i.status.host_ip
        try:
            container_id = i.status.container_statuses[0].container_id.split("//")[1][
                0:12
            ]
        except:
            container_id = ""
        try:
            cpu = i.spec.containers[0].resources.limits["cpu"]
            memory = i.spec.containers[0].resources.limits["memory"]
        except:
            cpu = 32
            memory = 64
        data = pd.DataFrame(
            [
                {
                    "microservice": pod_name,
                    "host_ip": host_ip,
                    "pod_ip": pod_ip,
                    "container_id": container_id,
                    "cpu": cpu,
                    "mem": memory,
                }
            ]
        )
        if pod_name in target_ms_set:
            target_data = pd.concat([target_data, data])
        if pod_name in parent_ms_set:
            parent_data = pd.concat([parent_data, data])
    return parent_data, target_data


def generate_if_data(
    priority: pd.DataFrame, parent_data: pd.DataFrame, target_data: pd.DataFrame
):
    left = (
        target_data.merge(priority, on="microservice")
        .rename(
            columns={
                "pod_ip": "dest_ip",
                "microservice": "dest_ms",
                "container_id": "dest_container",
                "host_ip": "dest_host",
            }
        )
        .drop(columns=["cpu", "mem", "service"])
    )
    right = parent_data[["microservice", "pod_ip"]].rename(
        columns={"microservice": "parent", "pod_ip": "src_ip"}
    )
    rules = left.merge(right, on="parent")[
        ["dest_host", "dest_container", "src_ip", "priority"]
    ]
    virtual_if_count: pd.DataFrame = (
        rules.groupby("dest_host")
        .apply(lambda x: len(x.groupby("dest_container")))
        .to_frame()
        .reset_index()
        .rename(columns={0: "count", "dest_host": "host"})
    )
    return rules, virtual_if_count


def generate_scripts(virtual_if_count: pd.DataFrame, rules: pd.DataFrame, data_path):
    result_path = f"{data_path}/onlineSchedulingResult/scripts"
    os.system(f"rm -rf {result_path}")
    os.system(f"mkdir -p {result_path}")

    for _, row in virtual_if_count.iterrows():
        host = str(row["host"])
        if_count = int(row["count"])
        clear_lines = ""
        # Config number of virtual interface
        lines = f"# Config number of virtual interface\n"
        lines += f"modprobe ifb numifbs={if_count}\n"
        lines += f'echo "Set numifbs to {if_count}"\n'
        host_rules = rules.loc[rules["dest_host"] == host]
        for container_index, container_id in enumerate(
            host_rules["dest_container"].unique().tolist()
        ):
            container_rules = host_rules.loc[rules["dest_container"] == container_id]
            ifb = f"ifb{container_index}"
            # Get container's network interface
            lines += f"# Get container's network interface\n"
            lines += f'echo "Working on container {container_index}"\n'
            lines += f"NET=`docker exec -i {container_id} bash -c 'cat /sys/class/net/eth*/iflink'`\n"
            lines += f"NET=`echo $NET|tr -d '\\r'`\n"
            lines += f"VETH=`grep -l $NET /sys/class/net/veth*/ifindex`\n"
            lines += f"VETH=`echo $VETH|sed -e 's;^.*net/\(.*\)/ifindex$;\\1;'`\n"
            lines += f"VETH=`echo $VETH|tr -d '\\r'`\n"
            lines += f'echo "Get container\'s network interface OK"\n'
            # Create virtual interface
            lines += f"# Create virtual interface\n"
            lines += f"ip link delete {ifb}\n"
            lines += f"ip link add {ifb} type ifb\n"
            lines += f"ip link set {ifb} up\n"
            lines += f'echo "Create network virtual interface OK"\n'
            # Redirect to virtual interface
            lines += f"# Redirect to virtual interface\n"
            lines += f"tc qdisc del dev $VETH handle ffff: ingress\n"
            lines += f"tc qdisc add dev $VETH handle ffff: ingress\n"
            lines += f"tc filter add dev $VETH parent ffff: \\\n"
            lines += f"    protocol ip u32 match u32 0 0 \\\n"
            lines += f"    flowid 1:1 action mirred egress redirect dev {ifb}\n"
            lines += f'echo "Add redirection OK"\n'
            # Add root
            lines += f"# Add root\n"
            lines += f"tc qdisc add dev {ifb} root handle 1:0 htb default 10\n"
            lines += f'echo "Add root OK"\n'
            # Add classes
            lines += f"# Add classes\n"
            for class_index, (_, rule) in enumerate(container_rules.iterrows()):
                src = str(rule["src_ip"]) + "/32"
                priority = int(rule["priority"])
                lines += f"tc class add dev {ifb} parent 1:0 classid 1:{class_index + 1} htb prio {priority} rate 10Mbit\n"
                lines += f"tc filter add dev {ifb} protocol ip parent 1:0 u32 match ip src {src} flowid 1:{class_index + 1}\n"
            lines += f'echo "Add classes OK"\n'

            # Clear
            clear_lines += f"NET=`docker exec -i {container_id} bash -c 'cat /sys/class/net/eth*/iflink'`\n"
            clear_lines += f"NET=`echo $NET|tr -d '\\r'`\n"
            clear_lines += f"VETH=`grep -l $NET /sys/class/net/veth*/ifindex`\n"
            clear_lines += f"VETH=`echo $VETH|sed -e 's;^.*net/\(.*\)/ifindex$;\\1;'`\n"
            clear_lines += f"VETH=`echo $VETH|tr -d '\\r'`\n"
            clear_lines += "tc qdisc del dev $VETH handle ffff: ingress\n"
            clear_lines += f"ip link delete {ifb}\n"
        lines += f'echo "Priority implemented"\n'
        with open(f"{result_path}/{host}-tc.sh", "w") as f:
            f.write(lines)
        with open(f"{result_path}/{host}-clean.sh", "w") as f:
            f.write(clear_lines)


def deploy_vifs(data_path):
    scripts_path = f"{data_path}/onlineSchedulingResult/scripts"
    scripts = [x for x in os.listdir(scripts_path) if x[-5:] == "tc.sh"]
    for script in scripts:
        host = script.split("-")[0]
        script_path = f"{scripts_path}/{script}"
        remote_execute_script(host, script_path, script)


def clear_vifs(data_path):
    scripts_path = f"{data_path}/onlineSchedulingResult/scripts"
    try:
        scripts = [x for x in os.listdir(scripts_path) if x[-8:] == "clean.sh"]
        for script in scripts:
            host = script.split("-")[0]
            script_path = f"{scripts_path}/{script}"
            remote_execute_script(host, script_path, script)
    except:
        print("No vifs needed to be clean")


def remote_execute_script(host, script_path, script):
    user = "root"
    os.system(f"scp {script_path} {user}@{host}:/tmp")
    os.system(f"ssh {user}@{host} bash /tmp/{script}")
    os.system(f"ssh {user}@{host} rm /tmp/{script}")
