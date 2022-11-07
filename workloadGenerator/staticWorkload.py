import os
import re
import subprocess
from typing import List

class StaticWorkloadGenerator:
    def __init__(self, threadNum, connectionNum, duration, reqFreq, wrkPath, scriptPath, url):
        """Config a static workload generator with given arguments.

        Args:
            threadNum (int): Number of threads each client uses
            connectionNum (int): Number of connections each client has
            duration (int): Testing duration with units "s"
            reqFreq (int): Number of requests sent per second
            wrkPath (str): The location of runnable wrk file
            scriptPath (str): The location of lua script
            url (str): The entry point of the test
        """
        self.wrkPath = wrkPath
        self.args = f"-D exp -t{threadNum} -c{connectionNum} -d{duration} " + \
            f"-R{reqFreq} -L -s {scriptPath}"
        self.url = url

    def generateWorkload(self, testName, clientNum):
        """Used to generate workload to test APP, service or microservice

        Args:
            testName (str): A unique test name, will be used as the folder name to store result
            clientNum (int): Number of concurrent clients sending requests
        """
        workload = f"{self.wrkPath} {self.args} {self.url} &"
        resultPath = f"tmp/wrkResult"

        if not os.path.isdir(resultPath):
            os.system(f"mkdir -p {resultPath}")
        os.system(f"rm -rf {resultPath}/{testName}")
        processes: List[subprocess.Popen] = []
        for _ in range(clientNum):
            proc = subprocess.Popen(workload, stdout=subprocess.PIPE, shell=True)
            processes.append(proc)
        for proc in processes:
            (out, _) = proc.communicate()
            with open(f"{resultPath}/{testName}", "a") as file:
                match = re.search(r"(\d+)\srequests", out.decode("utf-8"))
                file.write(match.group(1) + "\n")
            with open(f"{resultPath}/log", "a") as file:
                file.write(out.decode("utf-8") + "\n")