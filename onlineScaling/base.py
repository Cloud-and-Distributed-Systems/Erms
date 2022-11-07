import os
import pickle
import pandas as pd
from typing import Dict
from offlineProfiling.ermsModels import ErmsModel
from onlineScaling.Span import Span
from utils.files import load_model


class LatencyTargetComputation:
    def __init__(self, data_path):
        self.data_path = data_path
        self.result_path = f"{data_path}/scalingResult"
        os.system(f"mkdir -p {self.result_path}")
        self.root_spans: Dict[str, Span] = None
        self.db_data = None

    def priority_main(
        self, cpu_inter, mem_inter, workloads, slas, services=None
    ) -> None:
        pass

    def construct_graph(self, target_service=None):
        relation_df = pd.read_csv(
            f"{self.data_path}/offlineTestResult/spanRelationships.csv"
        )
        # To fix 1 will be read as 1.0
        relation_df = relation_df.astype({"tag": "str"})
        relation_df["tag"] = relation_df["tag"].str.replace(
            r"^(\d+).0", lambda x: x.group(1)
        )
        root_spans: Dict[str, Span] = {}
        for service in relation_df["service"].unique().tolist():
            if target_service is not None and service not in target_service:
                continue
            # Record all spans
            data = relation_df.loc[relation_df["service"] == service]
            first_lvl = data.loc[data["tag"].str.find(".") == -1]
            root_span = Span(
                first_lvl.iloc[0]["parentMS"], first_lvl.iloc[0]["parentOperation"], 1
            )

            def dfs(lvl: pd.DataFrame, parent: Span):
                for _, edge in lvl.iterrows():
                    tag = edge["tag"]
                    tag_reg = "^" + tag.replace(".", "\.") + ".\d+$"
                    child = Span(edge["childMS"], edge["childOperation"], edge["step"])
                    parent.append_child(child)
                    next_lvl = data.loc[data["tag"].str.match(tag_reg)]
                    dfs(next_lvl, child)

            dfs(first_lvl, root_span)

            root_spans[service] = root_span
        self.root_spans = root_spans

    def get_df(self):
        result = pd.DataFrame()
        for service, root_span in self.root_spans.items():
            df = root_span.to_df()
            df.insert(0, "service", [service] * len(df))
            result = pd.concat([result, df])
        return result.reset_index().drop(columns="index")

    def update_sharing_ms_models(self, models):
        root_spans_df = self.get_certain_type_df()
        for _, grp in root_spans_df.groupby("microservice"):
            grp = pd.DataFrame(grp).drop_duplicates(subset=["service", "microservice"])
            if len(grp) == 1:
                continue
            max_avg_slope = 0
            # select model that has the max average slope
            for _, row in grp.iterrows():
                if row["microservice"] not in models[row["service"]]:
                    continue
                ms_model = models[row["service"]][row["microservice"]]
                avg_slope = (ms_model["a1"] + ms_model["a2"]) / 2
                if avg_slope > max_avg_slope:
                    max_avg_slope = avg_slope
                    new_model = ms_model
            for _, row in grp.iterrows():
                models[row["service"]][row["microservice"]] = new_model
        for service, root_span in list(self.root_spans.items()):
            root_span.assign_model(models[service])
            root_span.merge_model()

    def generate_result(self):
        latency_target_by_ms = self.get_certain_type_df()
        container_by_ms = (
            latency_target_by_ms.groupby("microservice")["container"]
            .max()
            .reset_index()
        )
        return latency_target_by_ms, container_by_ms

    def save_to_csv(self, latency_target_file_name, container_file_name):
        latency_target_by_ms, container_by_ms = self.generate_result()
        latency_target_by_ms.to_csv(
            f"{self.result_path}/{latency_target_file_name}", index=False
        )
        container_by_ms.to_csv(f"{self.result_path}/{container_file_name}", index=False)
        return latency_target_by_ms, container_by_ms

    def process_db_data(self):
        if self.db_data != None:
            return self.db_data
        try:
            with open(f"{self.result_path}/db.temp", "rb") as file:
                self.db_data = pickle.load(file)
                return self.db_data
        except:
            pass
        db_data = pd.read_csv(f"{self.data_path}/offlineTestResult/db.csv")
        db_data = db_data.assign(
            identifier=db_data["parentMS"] + "|" + db_data["parentOperation"]
        )
        db_data = {
            x[0]: {
                y[0]: {z[0]: z[1].quantile(0.95)[0] for z in y[1].groupby("childMS")}
                for y in x[1].groupby("identifier")
            }
            for x in db_data.groupby("service")
        }
        self.db_data = db_data
        with open(f"{self.result_path}/db.temp", "wb") as file:
            pickle.dump(db_data, file)
        return db_data

    def get_certain_type_df(self) -> pd.DataFrame:
        pass

    def clear_db_cache(self):
        os.system(f"rm -rf {self.result_path}/db.temp")
        self.db_data = None


def compute_model(
    latency_models: pd.DataFrame, inflection_models: pd.DataFrame, cpu_inter, mem_inter
):
    models_dict = {}
    for _, model_record in latency_models.iterrows():
        model_path = model_record["modelFile"]
        service = model_record["service"]
        ms = model_record["microservice"]
        # Add slope and bias data
        model_type = model_record["type"]
        model = load_model(model_path)
        slope = model.predict_slope(cpu_inter, mem_inter)
        bias = model.get_bias()
        if service not in models_dict:
            models_dict[service] = {}
        if ms not in models_dict[service]:
            models_dict[service][ms] = {}
        if model_type == "beforeInflection":
            models_dict[service][ms]["a1"] = slope
            models_dict[service][ms]["b1"] = bias
        else:
            models_dict[service][ms]["a2"] = slope
            models_dict[service][ms]["b2"] = bias
        # Add inflection data
        tag = f"{service}_{ms}"
        if tag in inflection_models["tag"]:
            model_path = inflection_models.loc[inflection_models["tag"] == tag].iloc[0][
                "modelFile"
            ]
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            inflection = model.predict([[cpu_inter, mem_inter]])[0]
            models_dict[service][ms]["inflection"] = inflection
        else:
            models_dict[service][ms]["inflection"] = 0

    for service in models_dict:
        if "nginx" in models_dict[service]:
            models_dict[service]["nginx"] = {
                "a1": 0.00001,
                "a2": 0.00002,
                "b1": 0,
                "b2": 0,
                "inflection": 0,
            }
        if "nginx-web-server" in models_dict[service]:
            models_dict[service]["nginx-web-server"] = {
                "a1": 0.00001,
                "a2": 0.00002,
                "b1": 0,
                "b2": 0,
                "inflection": 0,
            }
        if "frontend" in models_dict[service]:
            models_dict[service]["frontend"] = {
                "a1": 0.00001,
                "a2": 0.00002,
                "b1": 0,
                "b2": 0,
                "inflection": 0,
            }
    return models_dict
