import os
import yaml
from kubernetes import utils, config, client

import utils.deploymentEditor as editor
from utils.others import wait_deletion, wait_deployment


class Deployer:
    def __init__(self, namespace, cpuSize, memorySize, nodes, yamlRepo, app_img):
        """Initializing a deployer for an certain APP

        Args:
            namespace (str): Namespace used to deploy pods
            cpuSize (str): CPU limitation
            memorySize (str): Memory limitation
            nodes (list[str]): List of nodes going to be used
            yamlRepo (str): Path to the folder that contains yaml files
        """
        self.namespace = namespace
        self.pod_spec = {"mem_size": memorySize, "cpu_size": cpuSize}
        self.nodes = nodes
        self.yamlRepo = yamlRepo
        self.tmpYamlRepo = f"tmp/testing-{namespace}"
        self.application_img = app_img
        # self.createNamespace()

    def full_init(self, app, infra_nodes, port):
        non_test_yamls = editor.read_all_yaml(f"{self.yamlRepo}/non-test")
        with open("yamlRepository/templates/deploymentAffinity.yaml", "r") as file:
            non_test_node_affinity = yaml.load(
                file.read().replace("%%%", f'[{", ".join(infra_nodes)}]'),
                yaml.CLoader,
            )

        path = "metadata.namespace"
        value = self.namespace
        non_test_yamls = editor.insert_to_python_objs(path, value, non_test_yamls, None)

        path = "spec.template.spec.affinity"
        value = non_test_node_affinity
        non_test_yamls = editor.insert_to_python_objs(path, value, non_test_yamls)

        path = "spec.template.spec.containers[0].imagePullPolicy"
        value = "IfNotPresent"
        non_test_yamls = editor.insert_to_python_objs(path, value, non_test_yamls)

        tmp_infra_path = f"tmp/infra-{self.namespace}"
        os.system(f"rm -rf {tmp_infra_path}")
        os.system(f"mkdir {tmp_infra_path}")
        editor.save_all_yaml(tmp_infra_path, non_test_yamls)
        # Deploy non_test_yamls
        delete_by_yaml(tmp_infra_path)
        deploy_by_yaml(tmp_infra_path, True, self.namespace)
        # Deploy test_yamls
        self.redeploy_app()
        # Import data
        if app == "social":
            from scripts.socialNetwork.init import main

            main(port=port)
        elif app == "media":
            from scripts.mediaMicroservice.write_movie_info import main

            main(server_address=f"http://localhost:{port}")
            from scripts.mediaMicroservice.register_movies_and_users import main

            main(server_address=f"http://localhost:{port}")

    def redeploy_app(self, containers=None):
        self.delete_app()
        self.deploy_app(containers)

    def deploy_app(self, containers=None):
        containers = containers if containers is not None else {}
        os.system(f"rm -rf {self.tmpYamlRepo}")
        os.system(f"mkdir -p {self.tmpYamlRepo}")

        # Read all yaml files
        yaml_list = editor.read_all_yaml(f"{self.yamlRepo}/test")
        yaml_list = editor.base_yaml_preparation(
            yaml_list, self.namespace, self.application_img, self.pod_spec
        )
        yaml_list = editor.assign_affinity(yaml_list, self.nodes)
        yaml_list = editor.assign_containers(yaml_list, containers)
        # Change image pull policy
        path = "spec.template.spec.containers[0].imagePullPolicy"
        value = "IfNotPresent"
        yaml_list = editor.insert_to_python_objs(path, value, yaml_list)
        # Save to the tmp folder
        editor.save_all_yaml(self.tmpYamlRepo, yaml_list)

        deploy_by_yaml(self.tmpYamlRepo, True, self.namespace)
    
    def delete_app(self, wait=False):
        delete_by_yaml(self.tmpYamlRepo, wait, self.namespace)

    def deployFromYaml(yamlPath, namespace):
        """Equal to: kubectl apply -f {yamlPath} -n {namespace}

        Args:
            yamlPath (str): Path to the yaml file
            namespace (str): Target namespace
        """
        config.load_kube_config()
        apiClient = client.ApiClient()
        utils.create_from_yaml(apiClient, yamlPath, namespace=namespace)

    def deleteDeployByNameInNamespace(name, namespace):
        """Equal to: kubectl delete deploy {name} -n {namespace}

        Args:
            name (str): Deployment name
            namespace (str): Namespace
        """
        try:
            config.load_kube_config()
            v1Client = client.AppsV1Api()
            v1Client.delete_namespaced_deployment(name, namespace)
        except:
            # If no deployment found then pass
            pass

    def createNamespace(self):
        """Equal to: kubectl create ns {namespace}

        Args:
            namespace (str): namespace
        """
        config.load_kube_config()
        coreV1Client = client.CoreV1Api()
        metadata = client.V1ObjectMeta(name=self.namespace)
        body = client.V1Namespace(api_version="v1", kind="Namespace", metadata=metadata)
        try:
            coreV1Client.create_namespace(body=body)
        except:
            # If namespace exist then pass
            pass


config.load_kube_config()


def deploy_by_yaml(folder, wait=False, namespace=None, timeout=300):
    api_client = client.ApiClient()
    for file in [
        x for x in os.listdir(folder) if x[-5:] == ".yaml" or x[-4:] == ".yml"
    ]:
        utils.create_from_yaml(api_client, f"{folder}/{file}")
    if wait:
        if namespace is None:
            raise BaseException("No namespace spcified")
        wait_deployment(namespace, timeout)


def delete_by_yaml(folder, wait=False, namespace=None, timeout=300, display=False):
    if display:
        os.system(f"kubectl delete -Rf {folder}")
    else:
        os.system(f"kubectl delete -Rf {folder} >/dev/null")
    if wait:
        if namespace is None:
            raise BaseException("No namespace spcified")
        wait_deletion(namespace, timeout)


def apply_by_yaml(folder, wait=False, namespace=None, timeout=300, display=False):
    if display:
        os.system(f"kubectl apply -Rf {folder}")
    else:
        os.system(f"kubectl apply -Rf {folder} >/dev/null")
    if wait:
        if namespace is None:
            raise BaseException("No namespace spcified")
        wait_deployment(namespace, timeout)
