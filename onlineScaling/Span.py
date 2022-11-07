from math import ceil, sqrt
import textwrap
from typing import Dict, List

import pandas as pd


class Span:
    def __init__(self, microservice, operation, step, **kargs):
        self.microservice = microservice
        self.operation = operation
        self.identifier = f"{self.microservice}|{self.operation}"
        self.step = step
        self.children: Dict[int, List[Span]] = {}

        self.model = {"a1": 0.000001, "b1": 0, "a2": 0.000001, "b2": 0, "inflection": 0}
        self.step_models = {}
        self.db_bias = 0
        self.merged_model = None
        self.latency_target = None
        # model_type has two possible values: "before" and "after"
        # "before": Before the inflection, will use parameters a1 and b1
        # "after": After the inflection, will use parameters a2 and b2
        self.model_type = "after"
        self.fixed_model_type = None
        self.workload = None
        self.containers = None
        self.data = kargs

    def append_child(self, child: "Span"):
        if child.step not in self.children:
            self.children[child.step] = []
        self.children[child.step].append(child)

    def remove_repeat(self, exist_ms=None):
        if exist_ms is None:
            exist_ms = []
        self.identifier = self.microservice
        exist_ms.append(self.microservice)
        to_removed = []
        for _, parallel_children in self.children.items():
            for index, span in enumerate(parallel_children):
                if span.microservice in exist_ms:
                    to_removed.append(span)
                    for child_index, child in enumerate(span.children):
                        self.children.insert(index + child_index + 1, child)
                    continue
                span.remove_repeat(exist_ms)
            for span in to_removed:
                parallel_children.remove(span)

    def assign_model(self, models):
        if self.microservice in models:
            self.model = models[self.microservice]
            self.model["inflection"] = (self.model["b2"] - self.model["b1"]) / (
                self.model["a1"] - self.model["a2"]
            )
            if self.model["inflection"] < 0:
                self.model["inflection"] = models[self.microservice]["inflection"]
        for _, parallel_children in self.children.items():
            for child in parallel_children:
                child.assign_model(models)

    def assign_workload(self, workload):
        if self.microservice in workload:
            self.workload = workload[self.microservice]
        else:
            self.workload = workload["default"]
        for _, parallel_children in self.children.items():
            for child in parallel_children:
                child.assign_workload(workload)

    def assign_db_bias(self, db_data: Dict[str, Dict[str, float]]):
        if self.identifier in db_data:
            to_removed_step = []
            for step, parallel_children in self.children.items():
                to_removed_child = []
                parallel_db_bias = 0
                for child in parallel_children:
                    for db_name in db_data[self.identifier].keys():
                        if db_name in child.operation:
                            to_removed_child.append(child)
                            parallel_db_bias = max(
                                db_data[self.identifier][db_name], parallel_db_bias
                            )
                for child in to_removed_child:
                    parallel_children.remove(child)
                if len(parallel_children) == 0:
                    to_removed_step.append(step)
                self.db_bias += parallel_db_bias
            for step in to_removed_step:
                self.children.pop(step, None)

        for _, parallel_children in self.children.items():
            for child in parallel_children:
                child.assign_db_bias(db_data)

    def merge_model(self):
        active_model = self.get_active_model()
        self.merged_model = active_model
        for step, parallel_children in self.children.items():
            child_models = []
            for child in parallel_children:
                child_models.append(child.merge_model())
            self.step_models[step] = parallel_merge(child_models)
            self.merged_model = sequential_merge(
                self.merged_model, self.step_models[step]
            )
        return self.merged_model

    def get_active_model(self):
        # Use different parameters based on model_type
        # See the defination of self.model_type
        if self.model_type == "before":
            active_model = {
                "a": self.model["a1"],
                "b": self.model["b1"] + self.db_bias,
                "workload": self.workload,
            }
        elif self.model_type == "after":
            active_model = {
                "a": self.model["a2"],
                "b": self.model["b2"] + self.db_bias,
                "workload": self.workload,
            }
        # todo its just a emergency patch
        if active_model["a"] < 0:
            active_model["a"] = 1
        return active_model
    
    def fix_model(self, fixed_ms_dict: Dict):
        if self.microservice in fixed_ms_dict:
            self.fixed_model_type = fixed_ms_dict[self.microservice]
        for step in self.children:
            for child in self.children[step]:
                child.fix_model(fixed_ms_dict)

    def compute_latency_target(self, latency_target):
        active_model = self.get_active_model()
        workload = self.workload if self.workload is not None else 1
        merged_workload = self.merged_model["workload"]
        merged_workload = merged_workload if merged_workload is not None else 1
        self.latency_target = (
            sqrt(active_model["a"] * workload)
            / sqrt(self.merged_model["a"] * merged_workload)
            * (latency_target - self.merged_model["b"])
            + active_model["b"]
        )
        for step, parallel_children in self.children.items():
            step_workload = self.step_models[step]["workload"]
            step_workload = step_workload if step_workload is not None else 1
            step_latency_target = (
                sqrt(self.step_models[step]["a"] * step_workload)
                / sqrt(self.merged_model["a"] * merged_workload)
                * (latency_target - self.merged_model["b"])
                + self.step_models[step]["b"]
            )
            for child in parallel_children:
                child.compute_latency_target(step_latency_target)
        self._update_model_type()

    def modify_specific_latency_target(self, latency_targets):
        if self.microservice in latency_targets:
            self.latency_target = latency_targets[self.microservice]
        for _, parallel_children in self.children.items():
            for child in parallel_children:
                child.modify_specific_latency_target(latency_targets)

    def _update_model_type(self):
        if self.fixed_model_type is not None:
            self.model_type = self.fixed_model_type
            return
        active_model = self.get_active_model()
        if (self.latency_target - active_model["b"]) / active_model["a"] <= self.model[
            "inflection"
        ]:
            self.model_type = "before"
        else:
            self.model_type = "after"

    def compute_container(self):
        if self.workload is None:
            raise BaseException(
                f"No workload for microservice {self.microservice} to calculate containers"
            )
        active_model = self.get_active_model()
        self.containers = ceil(
            active_model["a"]
            * self.workload
            / (self.latency_target - active_model["b"])
        )
        if self.containers <= 0:
            self.model_type = "before" if self.model_type == "after" else "after"
            active_model = self.get_active_model()
            self.containers = ceil(
                active_model["a"]
                * self.workload
                / (self.latency_target - active_model["b"])
            )
        for _, parallel_children in self.children.items():
            for child in parallel_children:
                child.compute_container()

    def to_str(
        self,
        data=[
            "latency_target",
            "inflection",
            "workload",
            "model",
            "merged_model",
            "container",
            "db_bias",
        ],
    ):
        active_model = self.get_active_model()
        optional_data = {
            "latency_target_str": f'Latency Target: {format(self.latency_target / 1000, ".2f")}ms'
            if isinstance(self.latency_target, float)
            else "NO LATENCY TARGET DATA",
            "inflection_str": f"Inflection: {self.model['inflection']}"
            if self.model["inflection"] != 0
            else "NO INFLECTION DATA",
            "workload_str": f"Workload: {self.workload}"
            if self.workload is not None
            else "NO WORKLOAD DATA",
            "model_str": (
                f"Model: {format(active_model['a'], '.2f')} X + "
                f"({format(active_model['b'], '.2f')}), {self.model_type}"
            ),
            "merged_model_str": (
                f"Merged Model: {format(self.merged_model['a'], '.2f')} X + "
                f"({format(self.merged_model['b'], '.2f')})"
            )
            if self.merged_model is not None
            else "NO MERGED MODEL DATA",
            "container_str": f"Containers: {self.containers}"
            if self.containers is not None
            else "NO CONTAINER DATA",
            "db_bias_str": f"Database Bias: {self.db_bias}",
        }

        final_str = f"{self.identifier}"
        for opt in data:
            final_str += " | " + optional_data[f"{opt}_str"]
        final_str += "\n"

        children_length = len(self.children.keys())
        if children_length != 0:
            for step_index, (_, parallel_children) in enumerate(self.children.items()):
                for child_index, child in enumerate(parallel_children):
                    if step_index + 1 != children_length or child_index + 1 != len(
                        parallel_children
                    ):
                        messages = child.to_str(data).split("\n")
                        child_id_str = f"├─{messages[0]}\n"
                        final_str += child_id_str + textwrap.indent(
                            "\n".join(messages[1:]), "│ ", lambda _: True
                        )
                    else:
                        messages = child.to_str(data).split("\n")
                        child_id_str = f"└─{messages[0]}\n"
                        final_str += child_id_str + textwrap.indent(
                            "\n".join(messages[1:]), "  ", lambda _: True
                        )
            return final_str
        else:
            return final_str[:-1]

    def to_df(self, parent: "Span" = None, df_type="erms"):
        active_model = self.get_active_model()
        df = pd.DataFrame(
            [
                {
                    "parent": parent.microservice if parent is not None else None,
                    "microservice": self.microservice,
                    "a": active_model["a"],
                    "b": active_model["b"],
                    "latency_target": self.latency_target,
                    "container": self.containers,
                    "type": self.model_type,
                    "workload": self.workload,
                    "inflection": self.model["inflection"],
                }
            ]
        )

        if df_type == "erms" and self.merged_model is not None:
            df = df.assign(
                merged_a=self.merged_model["a"], merged_b=self.merged_model["b"]
            )
        if df_type == "erms_computing":
            df = df.assign(a1=self.model["a1"])
            df = df.assign(a2=self.model["a2"])
            df = df.assign(b1=self.model["b1"])
            df = df.assign(b2=self.model["b2"])

        for _, parallel_children in self.children.items():
            for child in parallel_children:
                df = pd.concat([df, child.to_df(self, df_type)])
        return df

    def __hash__(self):
        return abs(hash(self.identifier)) % (10**8)

    def __eq__(self, obj):
        if isinstance(obj, Span):
            return self.identifier == obj.identifier
        else:
            return False

    def collapse_entrance(self):
        for _, parallel_children in self.children.items():
            to_removed = []
            to_append = []
            for child in parallel_children:
                if child.microservice == self.microservice:
                    to_removed.append(child)
                    child.collapse_entrance()
                    for _, parallel_grandson in child.children.items():
                        for grandson in parallel_grandson:
                            to_append.append(grandson)
            parallel_children.extend(to_append)
            for child in to_removed:
                parallel_children.remove(child)


def sequential_merge(model_1, model_2):
    if model_2 is None:
        return model_1
    workload_1 = model_1["workload"]
    workload_2 = model_2["workload"]
    if workload_1 == workload_2:
        workload = workload_1
        a = pow(sqrt(model_1["a"]) + sqrt(model_2["a"]), 2)
    else:
        workload = min(workload_1, workload_2)
        a = (
            pow(sqrt(model_1["a"] * workload_1) + sqrt(model_2["a"] * workload_2), 2)
            / workload
        )
    b = model_1["b"] + model_2["b"]

    return {"a": a, "b": b, "workload": workload}


def parallel_merge(models):
    if len(models) == 0:
        return None
    workload_list = [x["workload"] for x in models if x["workload"] is not None]
    if len(workload_list) == 0:
        workload = None
        a = sum([x["a"] for x in models])
        b = max([x["b"] for x in models])
    else:
        workload = min(workload_list)
        a = sum([x["a"] * x["workload"] / workload for x in models])
        b = max([x["b"] for x in models])
    return {"a": a, "b": b, "workload": workload}
