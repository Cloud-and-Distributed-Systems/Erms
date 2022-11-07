import os
import yaml
import utils.deploymentEditor as editor
from deployment.deployer import Deployer, deploy_by_yaml


class IdleInf:
    def __init__(self, cpu_size, mem_size, suffix, namespace="interference"):
        self.resource_limitation = {
            "requests": {"cpu": cpu_size, "memory": mem_size},
            "limits": {"cpu": cpu_size, "memory": mem_size}
        }
        self.namespace = namespace
        self.name_suffix = f"{suffix}-idle-inf"
    
    def _prepare_yaml(self, node, replicas):
        name = f"{self.name_suffix}-{node}"
        with open("yamlRepository/templates/deploymentAffinity.yaml", "r") as file:
            node_affinity = yaml.load(
                file.read().replace("%%%", f"[{node}]"), yaml.CLoader
            )
        yaml_file = editor.read_yaml("yamlRepository/templates/bestEffort.yaml")
        pairs = [
            ("metadata.name", name),
            ("metadata.namespace", self.namespace),
            ("spec.replicas", replicas),
            ("spec.template.spec.containers[0].resources", self.resource_limitation),
            ("spec.template.spec.affinity", node_affinity),
        ]
        for pair in pairs:
            yaml_file = editor.insert_to_python_objs(pair[0], pair[1], yaml_file)
        return yaml_file[0]
    
    def deploy_infs(self, nodes, replicas, wait=False):
        yaml_list = [self._prepare_yaml(node, replicas[node]) for node in nodes]
        os.system("rm -rf tmp/idleInfs")
        os.system("mkdir -p tmp/idleInfs")
        editor.save_all_yaml("tmp/idleInfs", yaml_list)
        deploy_by_yaml("tmp/idleInfs", wait, self.namespace)

    def delete_infs(self, nodes):
        for node in nodes:
            name = f"{self.name_suffix}-{node}"
            Deployer.deleteDeployByNameInNamespace(name, self.namespace)