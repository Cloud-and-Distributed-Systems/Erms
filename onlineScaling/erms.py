from copy import deepcopy
import pandas as pd
import onlineScaling.base as base


class ErmsBased(base.LatencyTargetComputation):
    def __init__(self, data_path):
        super().__init__(data_path)

    def priority_main(self, cpu_inter, mem_inter, workloads, slas, services=None):
        workloads = deepcopy(workloads)
        self.construct_graph(services)
        self.process_db_data()
        models = self.compute_model(cpu_inter, mem_inter)
        self.update_sharing_ms_models(models)
        self.init_span_data(workloads, models, self.db_data)
        self.calculate_latency_targets_and_containers(slas)
        if len(self.root_spans.keys()) > 1:
            self.priority_scheduling(workloads, slas)
        self.save_to_csv("ermsPrioLatencyTargetByMS.csv", "ermsPrioContainerByMS.csv")

    def non_priority_main(self, cpu_inter, mem_inter, workloads, slas, services=None):
        workloads = deepcopy(workloads)
        self.construct_graph(services)
        self.process_db_data()
        models = self.compute_model(cpu_inter, mem_inter)
        self.update_sharing_ms_models(models)
        self.init_span_data(workloads, models, self.db_data)
        self.calculate_latency_targets_and_containers(slas)
        if len(self.root_spans.keys()) > 1:
            self.non_priority_scheduling(workloads)
        self.save_to_csv(
            "ermsNonPrioLatencyTargetByMS.csv", "ermsNonPrioContainerByMS.csv"
        )

    def compute_model(self, cpu_inter, mem_inter):
        latency_models = pd.read_csv(f"{self.data_path}/fittingResult/ermsFull.csv")
        inflection_models = pd.read_csv(
            f"{self.data_path}/fittingResult/inflectionFitting.csv"
        )

        return base.compute_model(
            latency_models, inflection_models, cpu_inter, mem_inter
        )

    def calculate_latency_targets_and_containers(self, slas):
        for service, root_span in list(self.root_spans.items()):
            sla = slas[service]
            root_span.compute_latency_target(sla)
            root_span.merge_model()
            root_span.compute_latency_target(sla)
            df = root_span.to_df(df_type="erms_computing")
            while (df["latency_target"] < 0).any() or (
                df["latency_target"] < df["b1"]
            ).any():
                fixed_ms = (
                    df.loc[(df["latency_target"] < 0) | (df["latency_target"] < df["b1"])][
                        ["microservice", "type"]
                    ]
                    .drop_duplicates()
                    .assign(type="before")
                )
                fixed_ms = dict(zip(fixed_ms["microservice"], fixed_ms["type"]))
                root_span.fix_model(fixed_ms)
                root_span.merge_model()
                root_span.compute_latency_target(sla)
                df = root_span.to_df(df_type="erms_computing")
            root_span.compute_container()

    def init_span_data(self, workloads, models, db_data):
        for service, root_span in list(self.root_spans.items()):
            # root_span.collapse_entrance()
            root_span.assign_model(models[service])
            root_span.assign_workload(workloads[service])
            if service in db_data:
                root_span.assign_db_bias(db_data[service])
            root_span.merge_model()

    def priority_scheduling(self, workloads, slas):
        ms_latency_targets = self.get_certain_type_df()
        for _, data in ms_latency_targets.groupby("microservice"):
            data = pd.DataFrame(data).drop_duplicates(
                subset=["service", "microservice"]
            )
            if len(data) <= 1:
                continue
            cumulative_workload = 0
            for _, row in data.sort_values("latency_target").iterrows():
                service, ms = row[["service", "microservice"]]
                # Adjust workload
                original_workload = workloads[service]["default"]
                modified_workload = original_workload + cumulative_workload
                workloads[service][ms] = modified_workload
                cumulative_workload += original_workload

        for service, root_span in list(self.root_spans.items()):
            root_span.assign_workload(workloads[service])
            root_span.merge_model()

        self.calculate_latency_targets_and_containers(slas)

    def non_priority_scheduling(self, workloads):
        ms_latency_targets = self.get_certain_type_df()
        adjusted_latency_targets = {key: {} for key in self.root_spans.keys()}
        for _, data in ms_latency_targets.groupby("microservice"):
            data = pd.DataFrame(data).drop_duplicates(
                subset=["service", "microservice"]
            )
            if len(data) <= 1:
                continue
            new_latency_target = data["latency_target"].min()
            new_workload = data["workload"].sum()
            for _, row in data.iterrows():
                service = row["service"]
                ms = row["microservice"]
                workloads[service][ms] = new_workload
                adjusted_latency_targets[service][ms] = new_latency_target

        for service, root_span in list(self.root_spans.items()):
            root_span.assign_workload(workloads[service])
            root_span.modify_specific_latency_target(adjusted_latency_targets[service])
            root_span.compute_container()

    def get_certain_type_df(self):
        return super().get_df()
