import os
import re
from typing import Dict, Union
import pandas as pd
import yaml


def read_yaml(path):
    with open(path, "r") as file:
        return [x for x in yaml.load_all(file, Loader=yaml.CLoader) if x is not None]


def read_all_yaml(folder):
    yaml_files = [
        x for x in os.listdir(folder) if x[-5:] == ".yaml" or x[-4:] == ".yml"
    ]
    yaml_list = []
    for file_name in yaml_files:
        with open(f"{folder}/{file_name}", "r") as file:
            yaml_objs = yaml.load_all(file, Loader=yaml.CLoader)
            yaml_list.extend([x for x in yaml_objs if x is not None])
    return yaml_list


def insert_to_python_objs(
    path: str, value, yaml_list, target_kind="Deployment", key_path: str = None
):
    if key_path is not None:
        key_path = key_path.split(".")
    path = path.split(".")
    edited_yaml = []
    for yaml_obj in yaml_list:
        if target_kind is not None and yaml_obj["kind"] != target_kind:
            edited_yaml.append(yaml_obj)
            continue
        target = yaml_obj
        for index, prop in enumerate(path):
            if index == len(path) - 1:
                break
            search_result = re.search(r"(.*)\[(\d+)\]", prop)
            if search_result:
                # The user wants to insert/replace an item in a list
                prop = str(search_result.group(1))
                list_index = int(search_result.group(2))
                if prop in target:
                    exist_list_len = len(target[prop])
                    if list_index == exist_list_len:
                        new_target = {}
                        target[prop].append(new_target)
                        target = new_target
                    elif list_index < exist_list_len:
                        target = target[prop][list_index]
                    else:
                        raise Exception("A list will have empty entry after insertion!")
                elif list_index == 0:
                    new_target = {}
                    target[prop] = [new_target]
                    target = new_target
                else:
                    raise Exception("A list will have empty entry after insertion!")

            else:
                # The user is manipulating a dict
                if prop in target:
                    target = target[prop]
                else:
                    new_target = {}
                    target[prop] = new_target
                    target = new_target

        if key_path is not None:
            key = yaml_obj
            for prop in key_path:
                search_result = re.search(r"(.*)\[(\d+)\]", prop)
                if search_result:
                    prop = str(search_result.group(1))
                    list_index = int(search_result.group(2))
                    key = key[prop][list_index]
                else:
                    key = key[prop]
            if key in value:
                prop = path[-1]
                search_result = re.search(r"(.*)\[(\d+)\]", prop)
                if search_result:
                    prop = str(search_result.group(1))
                    list_index = int(search_result.group(2))
                    if prop in target:
                        exist_list_len = len(target[prop])
                        if list_index == exist_list_len:
                            target[prop].append(value[key])
                        elif list_index < exist_list_len:
                            target[prop][list_index] = value[key]
                        else:
                            raise Exception(
                                "A list will have empty entry after insertion!"
                            )
                    elif list_index == 0:
                        target[prop] = [value[key]]
                    else:
                        raise Exception("A list will have empty entry after insertion!")
                else:
                    target[path[-1]] = value[key]
        else:
            prop = path[-1]
            search_result = re.search(r"(.*)\[(\d+)\]", prop)
            if search_result:
                prop = str(search_result.group(1))
                list_index = int(search_result.group(2))
                if prop in target:
                    exist_list_len = len(target[prop])
                    if list_index == exist_list_len:
                        target[prop].append(value)
                    elif list_index < exist_list_len:
                        target[prop][list_index] = value
                    else:
                        raise Exception("A list will have empty entry after insertion!")
                elif list_index == 0:
                    target[prop] = [value]
                else:
                    raise Exception("A list will have empty entry after insertion!")
            else:
                target[path[-1]] = value
        edited_yaml.append(yaml_obj)
    return edited_yaml


def save_all_yaml(folder, yaml_list):
    os.system(f"rm -rf {folder}")
    os.system(f"mkdir -p {folder}")
    for yaml_obj in yaml_list:
        file_name = f'{yaml_obj["metadata"]["name"]}_{yaml_obj["kind"]}.yaml'
        with open(f"{folder}/{file_name}", "w") as file:
            yaml.dump(yaml_obj, file)


def base_yaml_preparation(yaml_list, namespace, app_img, pod_spec):
    path = "metadata.namespace"
    value = namespace
    yaml_list = insert_to_python_objs(path, value, yaml_list, None)

    path = "spec.template.spec.containers[0].image"
    value = {"APP_IMG": app_img}
    yaml_list = insert_to_python_objs(path, value, yaml_list, key_path=path)

    resource_limits = {
        "requests": {
            "memory": pod_spec["mem_size"],
            "cpu": pod_spec["cpu_size"],
        },
        "limits": {
            "memory": pod_spec["mem_size"],
            "cpu": pod_spec["cpu_size"],
        },
    }
    path = "spec.template.spec.containers[0].resources"
    value = resource_limits
    yaml_list = insert_to_python_objs(path, value, yaml_list)

    return yaml_list

def assign_containers(yaml_list, containers: Union[pd.DataFrame, Dict]):
    path = "spec.replicas"
    if isinstance(containers, Dict):
        value = containers
    else:
        value = dict(zip(containers["microservice"], containers["container"]))
    if "nginx-web-server" in value:
        value["nginx-thrift"] = value["nginx-web-server"]
    if "nginx" in value:
        value["nginx-web-server"] = value["nginx"]
    key_path = "metadata.name"
    yaml_list = insert_to_python_objs(path, value, yaml_list, key_path=key_path)

    return yaml_list

def assign_affinity(yaml_list, nodes):
    with open("yamlRepository/templates/deploymentAffinity.yaml", "r") as file:
        node_affinity = yaml.load(
            file.read().replace("%%%", f'[{", ".join(nodes)}]'),
            yaml.CLoader,
        )
    path = "spec.template.spec.affinity"
    value = node_affinity
    yaml_list = insert_to_python_objs(path, value, yaml_list)
    return yaml_list
