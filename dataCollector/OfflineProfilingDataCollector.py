from datetime import datetime
import multiprocessing
import os
from time import sleep
import traceback
from typing import Dict, List, Set
import json
import re
import pandas as pd
import requests
import utils.traceProcessor as t_processor
from utils.files import append_data
import utils.prometheus as prometheus_fetcher

pd.options.mode.chained_assignment = None


class OfflineProfilingDataCollector:
    def __init__(
        self,
        namespace,
        jaegerHost,
        entryPoint,
        prometheusHost,
        nodes,
        dataPath,
        duration=60,
        max_traces=1000,
        mointorInterval=1,
        max_processes=3
    ):
        """Initilizing an offline profiling data collector

        Args:
            namespace (str): Namespace
            duration (int): Duration of each round of test
            jaegerHost (str): Address to access jaeger, e.g. http://localhost:16686
            entryPoint (str): The entry point service of the test
            prometheusHost (str): Address to access Prometheus, similar to jaegerHost
            mointorInterval (str): Prometheus monitor interval
            nodes (list[str]): Nodes that will run test
            dataPath (str): Where to store merged data
            dataName (str): The name of the merged data
            cpuInterCpuSize (float | int): CPU limitation for CPU interference pod
            memoryInterMemorySize (str): Memory limiataion for memory interference pod
        """
        self.namespace = namespace
        self.duration = duration
        self.jaegerHost = jaegerHost
        self.entryPoint = entryPoint
        self.prometheusHost = prometheusHost
        self.monitorInterval = mointorInterval
        self.nodes = nodes
        self.max_traces = max_traces
        self.data_path = dataPath
        self.resultPath = f"{dataPath}/offlineTestResult"
        os.system(f"mkdir -p {self.resultPath}")
        manager = multiprocessing.Manager()
        self.relationDF = manager.dict()
        self.max_edges = manager.dict()
        self.max_processes = max_processes
        self.pool = multiprocessing.Pool(max_processes)

    def validation_collection_async(
        self,
        test_name,
        start_time,
        operation,
        service,
        repeat,
        data_path,
        no_nginx=False,
        no_frontend=False,
        **kwargs
    ):
        self.pool.apply_async(
            self.validation_collection,
            (
                test_name,
                start_time, 
                operation, 
                service, 
                repeat, 
                data_path,
                no_nginx,
                no_frontend,
            ),
            kwds=kwargs
        )

    def validation_collection(self, test_name, start_time, operation, service, repeat, data_path, no_nginx=False, no_frontend=False, **kwargs):
        os.system(f"mkdir -p {self.data_path}/{data_path}")
        self.log_file = f"log/{service}_validation.log"
        # Collect throughput data
        req_counter = self.collect_wrk_data(test_name)
        throughput = req_counter / self.duration
        _, span_data, trace_data = self.collect_trace_data(1500, start_time, operation, no_nginx, no_frontend)
        # Calculate mean latency of each microservice
        pod_latency, _ = self.process_span_data(span_data)
        ms_latency = pod_latency.groupby("microservice").mean().reset_index()
        # Get cpu usage or each microservice
        deployments = (
            pod_latency["pod"]
            .apply(lambda x: "-".join(str(x).split("-")[:-2]))
            .unique()
            .tolist()
        )
        # Remove empty string in deployments
        pod_cpu_usage = self.collect_cpu_usage(
            list(filter(lambda x: x, deployments)),
            start_time
        ).rename(columns={
            "usage": "cpuUsage",
            "deployment": "microservice"
        })
        ms_cpu_usage = (
            pod_cpu_usage
            .groupby("microservice")
            .mean()
            .reset_index()
        )
        # Merge ms data
        pod_latency = pod_latency.assign(
            service=service,
            repeat=repeat,
            throughput=throughput,
            **kwargs
        )
        ms_latency = ms_latency.assign(
            service=service, 
            repeat=repeat, 
            throughput=throughput,
            **kwargs
        )
        ms_latency = ms_latency.merge(ms_cpu_usage, on="microservice", how="left")
        pod_latency = pod_latency.merge(pod_cpu_usage, on=["microservice", "pod"], how="left")
        append_data(ms_latency, f"{self.data_path}/{data_path}/ms_latency.csv")
        append_data(pod_latency, f"{self.data_path}/{data_path}/pod_latency.csv")
        # Calculate mean trace latency of this test 
        trace_latency = trace_data[["traceLatency"]].assign(
            service=service, repeat=repeat, throughput=throughput,**kwargs
        )
        append_data(trace_latency, f"{self.data_path}/{data_path}/trace_latency.csv")
        print(
            f"P95: {format(trace_latency['traceLatency'].quantile(0.95) / 1000, '.2f')}ms, "
            f"throughput: {format(throughput, '.2f')}, "
            f"service: {service}, "
            f"repeat: {repeat}\n"
            f"data: {kwargs}"
        )

    def collect_wrk_data(self, file_name):
        """Get data from wrk file

        Returns:
            int: Accumulative counter of all lines in that wrk file
        """
        with open(f"tmp/wrkResult/{file_name}", "r") as file:
            lines = file.readlines()
            counter = 0
            for line in lines:
                counter += int(line)
            return counter

    def collect_trace_data(self, limit, start_time, operation=None, no_nginx=False, no_frontend=False):
        # Generate fetching url
        request_data = {
            "start": int(start_time * 1000000),
            "end": int((start_time + self.duration * 1000) * 1000000),
            "limit": limit,
            "service": self.entryPoint,
            "tags": '{"http.status_code":"200"}',
        }
        if operation is not None:
            request_data["operation"] = operation
        req = requests.get(f"{self.jaegerHost}/api/traces", params=request_data)
        self.write_log(f"Fetch latency data from: {req.url}")
        res = json.loads(req.content)["data"]
        if len(res) == 0:
            self.write_log(f"No traces are fetched!", "error")
            return False, None, None
        else:
            self.write_log(f"Number of traces: {len(res)}")
        # Record process id and microservice name mapping of all traces
        # Original headers: traceID, processes.p1.serviceName, processes.p2.serviceName, ...
        # Processed headers: traceId, p1, p2, ...
        service_id_mapping = (
            pd.json_normalize(res)
            .filter(regex="serviceName|traceID|tags")
            .rename(
                columns=lambda x: re.sub(
                    r"processes\.(.*)\.serviceName|processes\.(.*)\.tags",
                    lambda match_obj: match_obj.group(1)
                    if match_obj.group(1)
                    else f"{match_obj.group(2)}Pod",
                    x,
                )
            )
            .rename(columns={"traceID": "traceId"})
        )
        service_id_mapping = (
            service_id_mapping.filter(regex=".*Pod")
            .applymap(
                lambda x: [v["value"] for v in x if v["key"] == "hostname"][0]
                if isinstance(x, list)
                else ""
            )
            .combine_first(service_id_mapping)
        )
        spans_data = pd.json_normalize(res, record_path="spans")[
            [
                "traceID",
                "spanID",
                "operationName",
                "duration",
                "processID",
                "references",
                "startTime",
            ]
        ]
        spans_with_parent = spans_data[~(spans_data["references"].astype(str) == "[]")]
        root_spans = spans_data[(spans_data["references"].astype(str) == "[]")]
        root_spans = root_spans.rename(
            columns={
                "traceID": "traceId", 
                "startTime": "traceTime",
                "duration": "traceLatency"
            }
        )[["traceId", "traceTime", "traceLatency"]]
        spans_with_parent.loc[:, "parentId"] = spans_with_parent["references"].map(
            lambda x: x[0]["spanID"]
        )
        temp_parent_spans = spans_data[
            ["traceID", "spanID", "operationName", "duration", "processID"]
        ].rename(
            columns={
                "spanID": "parentId",
                "processID": "parentProcessId",
                "operationName": "parentOperation",
                "duration": "parentDuration",
                "traceID": "traceId",
            }
        )
        temp_children_spans = spans_with_parent[
            [
                "operationName",
                "duration",
                "parentId",
                "traceID",
                "spanID",
                "processID",
                "startTime",
            ]
        ].rename(
            columns={
                "spanID": "childId",
                "processID": "childProcessId",
                "operationName": "childOperation",
                "duration": "childDuration",
                "traceID": "traceId",
            }
        )
        # A merged data frame that build relationship of different spans
        merged_df = pd.merge(
            temp_parent_spans, temp_children_spans, on=["parentId", "traceId"]
        )

        merged_df = merged_df[
            [
                "traceId",
                "childOperation",
                "childDuration",
                "parentOperation",
                "parentDuration",
                "parentId",
                "childId",
                "parentProcessId",
                "childProcessId",
                "startTime",
            ]
        ]
        
        # Map each span's processId to its microservice name
        merged_df = merged_df.merge(service_id_mapping, on="traceId")
        merged_df = merged_df.merge(root_spans, on="traceId")
        merged_df = merged_df.assign(
            childMS=merged_df.apply(lambda x: x[x["childProcessId"]], axis=1),
            childPod=merged_df.apply(lambda x: x[f"{str(x['childProcessId'])}Pod"], axis=1),
            parentMS=merged_df.apply(lambda x: x[x["parentProcessId"]], axis=1),
            parentPod=merged_df.apply(
                lambda x: x[f"{str(x['parentProcessId'])}Pod"], axis=1
            ),
            endTime=merged_df["startTime"] + merged_df["childDuration"],
        )
        merged_df = merged_df[
            [
                "traceId",
                "traceTime",
                "startTime",
                "endTime",
                "parentId",
                "childId",
                "childOperation",
                "parentOperation",
                "childMS",
                "childPod",
                "parentMS",
                "parentPod",
                "parentDuration",
                "childDuration",
            ]
        ]
        if no_nginx:
            return True, merged_df, t_processor.no_entrance_trace_duration(merged_df, "nginx")
        elif no_frontend:
            return True, merged_df, t_processor.no_entrance_trace_duration(merged_df, "frontend")
        else:
            return True, merged_df, root_spans

    def construct_relationship(self, span_data: pd.DataFrame, max_edges: Dict, relation_df: Dict, service):
        if not service in max_edges:
            max_edges[service] = 0
        relation_result = t_processor.construct_relationship(
            span_data.assign(service=service),
            max_edges[service],
        )
        if relation_result:
            relation_df[service], max_edges[service] = relation_result
        pd.concat([x[1] for x in relation_df.items()]).to_csv(
            f"{self.resultPath}/spanRelationships.csv", index=False
        )

    def process_span_data(self, span_data: pd.DataFrame):
        db_data = pd.DataFrame()
        # for key_word in ["Mongo", "Redis", "Mmc", "Mem"]:
        #     dbs = span_data["childOperation"].str.contains(key_word)
        #     db_layer = span_data.loc[dbs]
        #     db_layer["childMS"] = key_word
        #     db_layer["childPod"] = key_word
        #     span_data = pd.concat([span_data.loc[~dbs], db_layer])
        #     db_data = pd.concat([
        #         db_data,
        #         db_layer[[
        #             "parentMS",
        #             "parentOperation", 
        #             "childMS", 
        #             "childOperation", 
        #             "childDuration"
        #         ]]
        #     ])
        # Calculate exact parent duration
        span_data = t_processor.exact_parent_duration(span_data, "merge")
        p95_df = t_processor.decouple_parent_and_child(span_data, 0.95)
        p50_df = t_processor.decouple_parent_and_child(span_data, 0.5)
        return p50_df.rename(columns={"latency": "median"}).merge(p95_df, on=["microservice", "pod"]), db_data

    def collect_data_async(self, test_data):
        self.pool.apply_async(
            self.collect_all_data,
            (test_data, self.max_edges, self.relationDF),
        )

    def collect_all_data(self, test_data, max_edges, relation_df):
        self.log_file = f"log/{test_data['service']}.log"
        try:
            req_counter = self.collect_wrk_data(test_data["test_name"])
            real_throughput = req_counter / self.duration
            self.write_log(
                f"Real Throughput: {real_throughput},"
                f"Target Throughput: {test_data['target_throughput']}"
            )
        except Exception:
            self.write_log("Collect wrk data failed!", "error")
            traceback.print_exc()
            return

        try:
            success, span_data, _ = self.collect_trace_data(self.max_traces, test_data["start_time"])
            if success:
                original_data = span_data.assign(
                    service=test_data["service"],
                    cpuInter=test_data["cpu_inter"],
                    memInter=test_data["mem_inter"],
                    targetThroughput=test_data["target_throughput"],
                    realThroughput=real_throughput,
                    repeat=test_data["repeat"],
                )
                self.construct_relationship(span_data, max_edges, relation_df, test_data["service"])
                latency_by_pod, db_data = self.process_span_data(span_data)
                append_data(db_data.assign(service=test_data["service"]), f"{self.resultPath}/db.csv")
                deployments = (
                    latency_by_pod["pod"]
                    .apply(lambda x: "-".join(str(x).split("-")[:-2]))
                    .unique()
                    .tolist()
                )
            else:
                return
        except Exception:
            self.write_log("Collect trace data failed!", "error")
            traceback.print_exc()
            return

        try:
            cpu_result = self.collect_cpu_usage(
                deployments, test_data["start_time"]
            ).rename(
                columns={"usage": "cpuUsage"}
            ).drop(columns="deployment")
        except Exception:
            self.write_log("Fetch CPU usage data failed!", "error")
            traceback.print_exc()
            return
        
        try:
            mem_result = self.collect_mem_usage(
                deployments, test_data["start_time"]
            ).rename(
                columns={"usage": "memUsage"}
            ).drop(columns="deployment")
        except Exception:
            self.write_log("Fetch memory usage data failed!", "error")
            traceback.print_exc()
            return

        try:
            latency_by_pod = (
                latency_by_pod
                .merge(cpu_result, on="pod", how="left")
                .merge(mem_result, on="pod", how="left")
            )
            original_data = original_data.merge(
                cpu_result, left_on="childPod", right_on="pod"
            ).merge(
                mem_result, left_on="childPod", right_on="pod"
            ).rename(
                columns={"cpuUsage": "childPodCpuUsage", "memUsage": "childPodMemUsage"}
            )
            original_data = original_data.merge(
                cpu_result, left_on="parentPod", right_on="pod"
            ).merge(
                mem_result, left_on="childPod", right_on="pod"
            ).rename(
                columns={"cpuUsage": "parentPodCpuUsage", "memUsage": "parentPodMemUsage"}
            )
            latency_by_pod = latency_by_pod.assign(
                repeat=test_data["repeat"],
                service=test_data["service"],
                cpuInter=test_data["cpu_inter"],
                memInter=test_data["mem_inter"],
                targetReqFreq=test_data["target_throughput"],
                reqFreq=real_throughput,
            )
            append_data(latency_by_pod, f"{self.resultPath}/latencyByPod.csv")
            append_data(original_data, f"{self.resultPath}/originalData.csv")
        except Exception:
            self.write_log("Merge all data failed!", "error")
            traceback.print_exc()

    def collect_cpu_usage(self, deployments, start_time):
        sleep(1)
        response = prometheus_fetcher.fetch_cpu_usage(
            self.prometheusHost, 
            self.namespace, 
            deployments, 
            start_time, 
            start_time + self.duration, 
            self.monitorInterval
        )
        self.write_log(f"Fetch CPU usage from: {response.url}")
        usage = response.json()
        cpu_result = pd.DataFrame(columns=["microservice", "pod", "usage"])
        if usage["data"] and usage["data"]["result"]:
            cpu_result = pd.DataFrame(data=usage["data"]["result"])
            cpu_result["pod"] = cpu_result["metric"].apply(lambda x: x["pod"])
            cpu_result["deployment"] = cpu_result["pod"].apply(
                lambda x: "-".join(x.split("-")[:-2])
            )
            cpu_result["usage"] = cpu_result["values"].apply(
                lambda x: max([float(v[1]) for v in x])
            )
            cpu_result = cpu_result[["deployment", "pod", "usage"]]
        return cpu_result

    def collect_mem_usage(self, deployments, start_time):
        sleep(1)
        response = prometheus_fetcher.fetch_mem_usage(
            self.prometheusHost, 
            self.namespace, 
            deployments, 
            start_time, 
            start_time + self.duration, 
            self.monitorInterval
        )
        self.write_log(f"Fetch memory usage from: {response.url}")
        usage = response.json()
        mem_result = pd.DataFrame(columns=["microservice", "pod", "usage"])
        if usage["data"] and usage["data"]["result"]:
            mem_result = pd.DataFrame(data=usage["data"]["result"])
            mem_result["pod"] = mem_result["metric"].apply(lambda x: x["pod"])
            mem_result["deployment"] = mem_result["pod"].apply(
                lambda x: "-".join(x.split("-")[:-2])
            )
            mem_result["usage"] = mem_result["values"].apply(
                lambda x: max([float(v[1]) for v in x])
            )
            mem_result = mem_result[["deployment", "pod", "usage"]]
        return mem_result

    def write_log(self, content, type="info"):
        with open(self.log_file, "a+") as file:
            current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            content = f"<{type}> <{current_time}> {content}\n"
            file.write(content)

    def wait_until_done(self):
        self.pool.close()
        self.pool.join()
        self.pool = multiprocessing.Pool(self.max_processes)
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)