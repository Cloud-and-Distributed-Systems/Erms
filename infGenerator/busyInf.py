import yaml
from deployment.deployer import Deployer, deploy_by_yaml
import os
import utils.deploymentEditor as editor
from utils.others import wait_deployment


class BusyInf:
    def __init__(self, nodes, cpuSize, memorySize, interferenceType, args, namespace="interference"):
        """Initialize a interference generator with certain type

        Args:
            nodes (list): Specify which nodes are going to be used to deploy pods
            cpuSize (str): Limitation of CPU
            memorySize (str): Limitation of memory
            interferenceType (str): "cpu" for CPU interference, "memory" for memory interference
            args (list): Additional arguments that need to pass to interferene docker
            namespace (str): Namespace used to deploy interference
        """
        self.resource_limits = {
            "requests": {"memory": memorySize, "cpu": cpuSize},
            "limits": {"memory": memorySize, "cpu": cpuSize},
        }
        self.interferenceType = interferenceType
        self.nodes = nodes
        self.args = args
        self.command = {"cpu": "/ibench/src/cpu", "memory": "/ibench/src/memCap"}[
            self.interferenceType
        ]
        self.namespace = namespace

    def generateInterference(self, replicas, wait=False):
        """Create a set of pods with the same CPU size and memory size on target nodes

        Args:
            replicas (int): Number of replicas of this CPU interference deployment
        """
        for node in self.nodes:
            self.generate_single_interference(node, replicas, False)
        if wait:
            wait_deployment(self.namespace, 300)

    def generate_single_interference(self, node, replicas, wait=False):
        """Create a set of pods with the same CPU size and memory size on a single node

        Args:
            node (str): Specify which node is going to use to deploy pods
            replicas (int): Number of replicas of this CPU interference deployment
        """
        name = f"{self.interferenceType}-interference-{node}"
        with open("yamlRepository/templates/deploymentAffinity.yaml", "r") as file:
            node_affinity = yaml.load(
                file.read().replace("%%%", f"[{node}]"), yaml.CLoader
            )
        yaml_file = editor.read_yaml("yamlRepository/templates/interference.yaml")
        pairs = [
            ("metadata.name", name),
            ("metadata.namespace", self.namespace),
            ("spec.replicas", replicas),
            ("spec.template.spec.containers[0].resources", self.resource_limits),
            ("spec.template.spec.containers[0].command[0]", self.command),
            ("spec.template.spec.containers[0].args", self.args),
            ("spec.template.spec.affinity", node_affinity),
        ]
        for pair in pairs:
            yaml_file = editor.insert_to_python_objs(pair[0], pair[1], yaml_file)
        os.system("rm -rf tmp/interference")
        os.system("mkdir -p tmp/interference")
        editor.save_all_yaml("tmp/interference", yaml_file)
        deploy_by_yaml("tmp/interference", wait, self.namespace)

    def clearAllInterference(self):
        """Delete all interference pod generated by this generator"""
        for node in self.nodes:
            name = f"{self.interferenceType}-interference-{node}"
            Deployer.deleteDeployByNameInNamespace(name, self.namespace)
