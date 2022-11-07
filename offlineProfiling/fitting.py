from math import ceil, floor, sqrt
import pickle
from statistics import mean
from typing import Dict, Tuple
from sklearn import tree
from configs import log
import pandas as pd
from box import Box
import os
from scipy.optimize import curve_fit
import random
import matplotlib.pyplot as plt

from offlineProfiling.ermsModels import ErmsModel, FittingBySlope as SlopeModel
from offlineProfiling.ermsModels import FullFitting as FullModel
from offlineProfiling.ermsModels import FittingUsage as UsageModel
from utils.files import append_data, dump_model


def linear(reqFreq, a, b):
    return a * reqFreq + b


class Fitting:
    def __init__(
        self,
        data_path,
        portion_of_data,
        range_dict,
        figure_path,
        cpu_usage_limits,
        min_of_max_cpu,
        throughput_error_rate,
        class_gap,
        replica_dict,
        target_services=None,
        draw_figures=False,
    ):
        self.accy = 0.3
        self.cpu_usage_limits = {
            "lower": cpu_usage_limits["lower"] * 100,
            "higher": cpu_usage_limits["higher"] * 100,
        }
        self.min_of_max_cpu = min_of_max_cpu * 100
        self.throughput_error_rate = throughput_error_rate
        self.class_gap = class_gap
        if target_services == "All":
            target_services = None

        self.result_path = f"{data_path}/fittingResult"
        self.figure_path = f"{figure_path}/fittingFigure"

        self.draw_figures = draw_figures
        self._prepare_data(data_path, target_services, replica_dict)
        self.portion_of_data = portion_of_data
        self.range_dict = range_dict
        self.inflection_points = None

    def erms_main(self):
        os.system(f"mkdir -p {self.result_path}/models/inflectionPrediction")
        os.system(f"mkdir -p {self.result_path}/models/latencyPrediction")
        os.system(f"mkdir -p {self.result_path}/models/usagePrediction")
        # Try to find the best inflection result
        inflection_data, updated_df = self.find_inflection()
        # bias_result, bias_models = self.fitting_bias(inflection_data)
        inflection_result, inflection_models = self.fittingInflection(inflection_data)
        self.save_inflection_related_result(
            inflection_data,
            updated_df,
            # bias_result,
            # bias_models,
            inflection_result,
            inflection_models,
        )

        self.usage_fitting()
        self.erms_model_full_fitting()
        self.low_usage_ms_model(self.low_usage_ms_df)

    def find_inflection(self):
        """Find the inflection point in a given range

        Returns: [
            "tag", "inflection", "a1", "b1", "a2", "b2",
            "accy", "cpuInter", "memInter", "success"
        ]
        """

        def test_in_all_range(data: pd.DataFrame, inflection_range):
            best_result = {"success": False}
            # Find the best inflection point in the given range
            for inflection in inflection_range:
                inflection = inflection / data.iloc[0]["replicas"]
                part_1 = data[data["reqFreq"] < inflection]
                part_2 = data[data["reqFreq"] >= inflection]
                # Size of data of part 1 and part 2 are different
                # So, assign their accuracy with different weight
                weight_1 = (
                    data.loc[data["reqFreq"] < inflection]["reqFreq"].count()
                    / data["reqFreq"].count()
                )
                weight_2 = 1 - weight_1
                try:
                    result_1 = self.microserviceLatencyReqFitting(part_1)
                    result_2 = self.microserviceLatencyReqFitting(part_2)
                except (TypeError, ValueError):
                    continue
                accy = weight_1 * result_1["accy"] + weight_2 * result_2["accy"]
                # Calculate gap between two functions, the smaller the gap, the better the result
                gap = abs(
                    linear(inflection, result_1["a"], result_1["b"])
                    - linear(inflection, result_2["a"], result_2["b"])
                )
                gap = max(1, gap)
                normalized_gap = gap / (data["latency"].max() - data["latency"].min())
                grades = accy / normalized_gap
                if result_2["a"] < 0 or result_1["a"] < 0:
                    grades = 0
                if "grades" not in best_result or grades > best_result["grades"]:
                    best_result["inflection"] = inflection
                    best_result["accy"] = accy
                    best_result["a1"] = result_1["a"]
                    best_result["b1"] = result_1["b"]
                    best_result["a2"] = result_2["a"]
                    best_result["b2"] = result_2["b"]
                    best_result["grades"] = grades
                    best_result["success"] = True
            best_result.pop("grades", None)
            return best_result

        def remove_noise(data: pd.DataFrame, best_inflection):
            inflection_latency = linear(
                best_inflection["inflection"],
                best_inflection["a1"],
                best_inflection["b1"],
            )
            part_1 = data.loc[data["reqFreq"] <= best_inflection["inflection"]]
            part_2 = data.loc[data["reqFreq"] >= best_inflection["inflection"]]
            predict_1 = linear(
                part_1["reqFreq"], best_inflection["a1"], best_inflection["b1"]
            )
            predict_2 = linear(
                part_2["reqFreq"], best_inflection["a2"], best_inflection["b2"]
            )
            accy_1 = 1 - abs((predict_1 - part_1["latency"]) / part_1["latency"])
            accy_2 = 1 - abs((predict_2 - part_2["latency"]) / part_2["latency"])
            inflection_latency_ratio_1 = abs(
                (predict_1 - part_1["latency"]) / inflection_latency
            )
            inflection_latency_ratio_2 = abs(
                (predict_2 - part_2["latency"]) / inflection_latency
            )
            return pd.concat(
                [
                    part_1.loc[
                        (accy_1 >= self.accy) | (inflection_latency_ratio_1 < 1.5)
                    ],
                    part_2.loc[
                        (accy_2 >= self.accy) | (inflection_latency_ratio_2 < 2.5)
                    ],
                ]
            )

        results = []
        intermediate_results = []
        intermediate_df = []
        updated_df = []
        tags = self.dataDF["tag"].unique().tolist()
        log.info(f"Total number of tags: {len(tags)}")
        for index, tag in enumerate(tags):
            log.info(f"Finding Inflection: {tag}, {len(tags) - index} tags left")
            (service, cpu, mem, ms) = tag.split("_")
            data = self.dataDF[self.dataDF["tag"] == tag]
            inflection_range = self.range_dict[service]
            intermediate = test_in_all_range(data, inflection_range)
            if not intermediate["success"]:
                results.append(
                    {
                        "tag": f"{service}_{ms}",
                        "cpuInter": cpu,
                        "memInter": mem,
                        "success": False,
                    }
                )
                log.warn("Fitting failed!", update=True, spinner=False)
                continue
            intermediate.update(
                {"tag": f"{service}_{ms}", "cpuInter": cpu, "memInter": mem}
            )
            intermediate_results.append(intermediate)
            intermediate_df.append(data)
            clean_data = remove_noise(data, intermediate)
            test_result = test_in_all_range(clean_data, inflection_range)
            if not test_result["success"]:
                results.append(
                    {
                        "tag": f"{service}_{ms}",
                        "cpuInter": cpu,
                        "memInter": mem,
                        "success": False,
                    }
                )
                log.warn("Fitting failed!", update=True, spinner=False)
                continue
            updated_df.append(clean_data)
            test_result.update(
                {"tag": f"{service}_{ms}", "cpuInter": cpu, "memInter": mem}
            )
            results.append(test_result)
            log.info(
                f"Fitting success, accy: {test_result['accy']}",
                update=True,
                spinner=False,
            )
        if self.draw_figures:
            self._drawInflectionFigure(
                pd.concat(intermediate_df),
                pd.DataFrame(intermediate_results),
                "intermediate",
            )
        return (pd.DataFrame(results), pd.concat(updated_df))

    def fittingInflection(
        self, inflection_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        result = pd.DataFrame(columns=["tag", "accy", "modelFile"])
        models = {}
        for tag, data in inflection_data.groupby("tag"):
            data = pd.DataFrame(data).loc[data["success"]]

            try:
                reg = tree.DecisionTreeRegressor(max_depth=3)
                reg = reg.fit(data[["cpuInter", "memInter"]], data["inflection"])
                predict = reg.predict(data[["cpuInter", "memInter"]])
                accy = (
                    1
                    - abs(pd.Series(predict) - data.reset_index()["inflection"])
                    / data.reset_index()["inflection"]
                ).mean()
                model_path = f"{self.result_path}/models/inflectionPrediction/{tag}.DT"
                models[tag] = {"path": model_path, "model": reg}
                params = pd.DataFrame(
                    {"tag": tag, "accy": accy, "modelFile": model_path}, index=[0]
                )
                result = pd.concat([result, params])
            except (ValueError, TypeError):
                pass
        return result, models

    def fitting_bias(self, inflection_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        result = []
        models = {}
        for tag, data in inflection_data.groupby("tag"):
            data = pd.DataFrame(data).loc[data["success"]]
            _, test_data = self.split_test_train(data, self.portion_of_data)

            try:
                reg = tree.DecisionTreeRegressor(max_depth=3)
                reg = reg.fit(data[["cpuInter", "memInter"]], data["b2"])
                predict = reg.predict(test_data[["cpuInter", "memInter"]])
                accy = (1 - abs((predict - test_data["b2"]) / test_data["b2"])).mean()
                model_path = f"{self.result_path}/models/biasPrediction/{tag}.DT"
                models[tag] = {"path": model_path, "model": reg}
                params = pd.DataFrame(
                    {"tag": tag, "accy": accy, "modelFile": model_path}, index=[0]
                )
                result.append(params)
            except (ValueError, TypeError):
                pass
        return pd.concat(result), models

    def save_inflection_related_result(
        self,
        inflection_data: pd.DataFrame,
        updated_df: pd.DataFrame,
        # bias_result: pd.DataFrame,
        # bias_models: Dict,
        inflection_result: pd.DataFrame,
        inflection_models: Dict,
    ):
        self.inflection_points = inflection_data
        self.dataDF = updated_df
        os.system(f"mkdir -p {self.result_path}")
        inflection_data.to_csv(f"{self.result_path}/inflection.csv", index=False)
        if self.draw_figures:
            self._drawInflectionFigure(updated_df, inflection_data)

        # os.system(f"mkdir -p {self.result_path}/models/biasPrediction")
        # bias_result.to_csv(f"{self.result_path}/biasPrediction.csv", index=False)
        # self.bias_models = {}
        # for tag, model_data in bias_models.items():
        #     self.bias_models[tag] = model_data["model"]
        #     with open(model_data["path"], "wb") as file:
        #         pickle.dump(model_data["model"], file)
        # os.system(f"mkdir -p {self.result_path}/models/inflectionPrediction")
        inflection_result.to_csv(
            f"{self.result_path}/inflectionFitting.csv", index=False
        )
        for _, model_data in inflection_models.items():
            dump_model(model_data["model"], model_data["path"])

    def microserviceLatencyReqFitting(self, data):
        result = Box({"a": 0, "b": 0, "accy": 0})
        popt, _ = curve_fit(linear, data["reqFreq"], data["latency"])
        result.a = popt[0]
        result.b = popt[1]
        result.accy = self.calulate_accuracy(
            data["reqFreq"], data["latency"], linear, result.a, result.b
        )
        return result

    def erms_model_full_fitting(self):
        os.system(f"mkdir -p {self.result_path}/models/latencyPrediction")
        if type(self.inflection_points) == type(None):
            inflection_data = self.find_inflection()
        else:
            inflection_data = self.inflection_points

        def fitting(data: pd.DataFrame):
            results = []
            before_data = data.loc[
                (data["reqFreq"] <= data["inflection"]) & (data["success"])
            ]
            after_data = data.loc[
                (data["reqFreq"] > data["inflection"]) & (data["success"])
            ]
            for index in range(2):
                part_data, model_type, suffix = [
                    (before_data, "beforeInflection", "before"),
                    (after_data, "afterInflection", "after"),
                ][index]
                train_data, test_data = self.split_test_train_by_inter(
                    part_data, self.portion_of_data
                )
                if len(train_data) < 4:
                    train_data = test_data
                model = FullModel()
                try:
                    model.train(
                        train_data["reqFreq"],
                        train_data["cpuInter"],
                        train_data["memInter"],
                        train_data["latency"],
                    )
                    accy = model.score(
                        test_data["reqFreq"],
                        test_data["cpuInter"],
                        test_data["memInter"],
                        test_data["latency"],
                    )
                except:
                    continue
                model_path = f"{self.result_path}/models/latencyPrediction/{data.iloc[0]['tag']}_{suffix}.full"
                dump_model(model, model_path)
                log.info(f"Accy {suffix} inflection: {accy}")
                results.append(
                    {"type": model_type, "accy": accy, "modelFile": model_path}
                )
            return pd.DataFrame(results)

        fitting_result = pd.DataFrame(
            columns=["service", "microservice", "type", "accy", "modelFile"]
        )
        tags = inflection_data["tag"].unique().tolist()
        for index, tag in enumerate(tags):
            log.info(f"Erms Full Fitting: {tag}, {len(tags) - index} tags left")
            service, ms = str(tag).split("_")
            latency_data = self.dataDF.loc[
                (self.dataDF["service"] == service)
                & (self.dataDF["microservice"] == ms)
            ]
            data = latency_data.drop(columns="tag").merge(
                inflection_data.loc[inflection_data["tag"] == tag].astype(
                    {"cpuInter": float, "memInter": float}
                ),
                on=["cpuInter", "memInter"],
                how="left",
            )
            ms_result = fitting(data).assign(service=service, microservice=ms)
            fitting_result = pd.concat([fitting_result, ms_result])

        append_data(fitting_result, f"{self.result_path}/ermsFull.csv")
        if self.draw_figures:
            os.system(f"mkdir -p {self.figure_path}/fullFitting")
            self._draw_erms_fitting_fig(fitting_result, "fullFitting")

    def erms_model_slope_fitting(self):
        os.system(f"mkdir -p {self.result_path}/models/latencyPrediction")
        if type(self.inflection_points) == type(None):
            inflection_data = self.find_inflection()
        else:
            inflection_data = self.inflection_points

        def fitting(data):
            train_data, test_data = self.split_test_train(data, self.portion_of_data)
            results = []

            for i in range(2):
                suffix, slope, bias, model_type = [
                    ("before", "a1", "b1", "beforeInflection"),
                    ("after", "a2", "b2", "afterInflection"),
                ][i]
                model = SlopeModel()
                model.train(
                    train_data["cpuInter"],
                    train_data["memInter"],
                    train_data[slope],
                    train_data[bias],
                )
                accy = model.score(
                    test_data["cpuInter"], test_data["memInter"], test_data[slope]
                )
                model_path = f"{self.result_path}/models/latencyPrediction/{data.iloc[0]['tag']}_{suffix}.slope"
                dump_model(model, model_path)
                log.info(f"Accy {suffix} inflection: {accy}")
                results.append(
                    {"type": model_type, "accy": accy, "modelFile": model_path}
                )

            return pd.DataFrame(results)

        fitting_result = pd.DataFrame(
            columns=["service", "microservice", "type", "accy", "modelFile"]
        )
        tag_list = inflection_data["tag"].unique().tolist()
        for index, tag in enumerate(tag_list):
            log.info(f"Erms Slope Fitting: {tag}, {len(tag_list) - index} tags left")
            service, ms = str(tag).split("_")
            data = inflection_data.loc[
                (inflection_data["tag"] == tag) & (inflection_data["success"])
            ]
            data = data.astype({"cpuInter": float, "memInter": float})
            ms_result = fitting(data).assign(service=service, microservice=ms)
            fitting_result = pd.concat([fitting_result, ms_result])

        fitting_result.to_csv(f"{self.result_path}/ermsSlope.csv")
        if self.draw_figures:
            os.system(f"mkdir -p {self.figure_path}/slopeFitting")
            self._draw_erms_fitting_fig(fitting_result, "slopeFitting")

    def erms_parameter_fitting(self):
        os.system(f"mkdir -p {self.result_path}/models/parametersPrediction")
        if type(self.inflection_points) == type(None):
            inflection_data = self.find_inflection()
        else:
            inflection_data = self.inflection_points

        parameters_data = []
        for tag, rows in self.dataDF.groupby("tag"):
            service, cpu_inter, mem_inter, ms = str(tag).split("_")
            inflection = inflection_data.loc[
                (inflection_data["tag"] == f"{service}_{ms}")
                & (inflection_data["cpuInter"] == cpu_inter)
                & (inflection_data["memInter"] == mem_inter)
            ].iloc[0]["inflection"]
            data = [
                rows.loc[rows["reqFreq"] <= inflection],
                rows.loc[rows["reqFreq"] >= inflection],
            ]
            for i in range(2):
                train_data = data[i]
                model_type = ["before", "after"][i]
                model = FullModel()
                try:
                    model.train(
                        train_data["reqFreq"],
                        train_data["cpuInter"],
                        train_data["memInter"],
                        train_data["latency"],
                    )
                except:
                    continue
                parameters_data.append(
                    {
                        "service": service,
                        "microservice": ms,
                        "cpuInter": float(cpu_inter),
                        "memInter": float(mem_inter),
                        "a": model.a,
                        "b": model.b,
                        "c": model.c,
                        "d": model.d,
                        "modelType": model_type,
                    }
                )

        result = pd.DataFrame()
        for (service, ms, model_type), rows in pd.DataFrame(parameters_data).groupby(
            ["service", "microservice", "modelType"]
        ):
            train_data, test_data = self.split_test_train_by_inter(
                rows, self.portion_of_data
            )
            try:
                reg = tree.DecisionTreeRegressor(max_depth=3)
                reg = reg.fit(
                    train_data[["cpuInter", "memInter"]],
                    train_data[["a", "b", "c", "d"]],
                )
                predict = reg.predict(test_data[["cpuInter", "memInter"]])
                predict = pd.DataFrame(predict, columns=["a", "b", "c", "d"])
                actual = test_data.reset_index()[["a", "b", "c", "d"]]
                accy_a, accy_b, accy_c, accy_d = (
                    1 - abs((predict - actual) / actual)
                ).mean()
                accy = mean([accy_a, accy_b, accy_c, accy_d])
                model_path = f"{self.result_path}/models/parametersPrediction/{service}_{ms}_{model_type}.DT"
                with open(model_path, "wb") as file:
                    pickle.dump(reg, file)
                result = pd.concat(
                    [
                        result,
                        pd.DataFrame(
                            {
                                "service": service,
                                "microservice": ms,
                                "accy": accy,
                                "accyA": accy_a,
                                "accyB": accy_b,
                                "accyC": accy_c,
                                "accyD": accy_d,
                                "modelType": model_type,
                                "modelFile": model_path,
                            },
                            index=[0],
                        ),
                    ]
                )
            except (ValueError, TypeError):
                pass

        result.to_csv(f"{self.result_path}/parameterPrediction.csv", index=False)

    def usage_fitting(self):
        os.system(f"mkdir -p {self.result_path}/models/usagePrediction")
        cpu_results = []
        mem_results = []
        for (service, ms), rows in self.dataDF.groupby(["service", "microservice"]):
            data = rows.dropna()
            cpu_model = UsageModel()
            cpu_model.train(data["reqFreq"], data["cpuUsage"])
            accy = cpu_model.score(data["reqFreq"], data["cpuUsage"])
            model_path = f"{self.result_path}/models/usagePrediction/{service}_{ms}.cpu"
            dump_model(cpu_model, model_path)
            cpu_results.append(
                {
                    "service": service,
                    "microservice": ms,
                    "accy": accy,
                    "modelFile": model_path,
                }
            )
            mem_model = UsageModel()
            mem_model.train(data["reqFreq"], data["memUsage"])
            accy = mem_model.score(data["reqFreq"], data["memUsage"])
            model_path = f"{self.result_path}/models/usagePrediction/{service}_{ms}.mem"
            dump_model(mem_model, model_path)
            mem_results.append(
                {
                    "service": service,
                    "microservice": ms,
                    "accy": accy,
                    "modelFile": model_path,
                }
            )
        pd.DataFrame(cpu_results).to_csv(
            f"{self.result_path}/cpuUsagePrediction.csv", index=False
        )
        pd.DataFrame(mem_results).to_csv(
            f"{self.result_path}/memUsagePrediction.csv", index=False
        )

    @staticmethod
    def dataSmoothing(data, column):
        """Doing smoothing on certain column of data

        Args:
            data (DataFrame): DataFrame
            column (str): Name of the column need to be smoothed
            n (int): Used to config smooth strength

        Returns:
            DataFrame: Smoothed DataFrame
        """
        newData = data.copy()
        newData = newData.assign(_temp=newData[column])
        newData = newData.drop(columns=column)
        newData = newData.assign(
            **{column: newData["_temp"].rolling(3).quantile(0.5, interpolation="lower")}
        )
        newData = newData.drop(columns="_temp")
        return newData.dropna()

    @staticmethod
    def split_test_train(data: pd.DataFrame, portion_of_train):
        columns = data.columns
        data = data.reset_index()
        train_data_indexes = random.sample(
            data.index.tolist(), int(len(data) * portion_of_train)
        )
        train_data = data[data.index.isin(train_data_indexes)]
        test_data = data[~data.index.isin(train_data_indexes)]
        return train_data[columns], test_data[columns]

    @staticmethod
    def split_test_train_by_inter(data: pd.DataFrame, portion_of_train):
        inter_grp = data.groupby(["cpuInter", "memInter"]).groups
        train_grp_indexes = random.sample(
            list(inter_grp.items()), int(portion_of_train * len(inter_grp))
        )
        train_data_indexes = [y for x in train_grp_indexes for y in x[1]]
        train_data = data.loc[data.index.isin(train_data_indexes)]
        test_data = data.loc[~data.index.isin(train_data_indexes)]
        return train_data, test_data

    @staticmethod
    def calulate_accuracy(test_x, test_y, model, *args):
        prediction = model(test_x, *args)
        accy = 1 - abs(prediction - test_y) / test_y
        return accy.quantile(0.5)

    def low_usage_ms_model(self, low_usage_ms_df: pd.DataFrame):
        os.system(f"mkdir -p {self.result_path}/models/latencyPrediction")
        low_usage_ms_df = (
            low_usage_ms_df.groupby(["service", "microservice"])["latency"]
            .quantile(0.95)
            .reset_index()
        )
        result = []
        for _, row in low_usage_ms_df.iterrows():
            service = row["service"]
            microservice = row["microservice"]

            model = FullModel(0, 0, 0.001, row["latency"])
            model_file = f"{self.result_path}/models/latencyPrediction/{service}_{microservice}_before.full"
            dump_model(model, model_file)
            result.append(
                {
                    "service": service,
                    "microservice": microservice,
                    "type": "beforeInflection",
                    "accy": 1,
                    "modelFile": model_file,
                }
            )
            model = FullModel(0, 0, 0.002, row["latency"])
            model_file = f"{self.result_path}/models/latencyPrediction/{service}_{microservice}_after.full"
            dump_model(model, model_file)
            result.append(
                {
                    "service": service,
                    "microservice": microservice,
                    "type": "afterInflection",
                    "accy": 1,
                    "modelFile": model_file,
                }
            )
        result = pd.DataFrame(
            result, columns=["service", "microservice", "type", "accy", "modelFile"]
        )
        result.to_csv(f"{self.result_path}/ermsFull.csv", index=False, mode="a", header=False)

    def _prepare_data(self, data_path, target_services, replica_dict):
        # reading data
        path = f"{data_path}/offlineTestResult/latencyByPod.csv"
        self.dataDF = pd.read_csv(path)
        if isinstance(target_services, list):
            self.dataDF = self.dataDF.loc[self.dataDF["service"].isin(target_services)]
        elif target_services is not None:
            self.dataDF = self.dataDF.loc[self.dataDF["service"] == target_services]
        # Remove unstable data, i.e. remove data that has maximum
        # cpu usage too high or too low
        # If data is recovered from old data, cpuUsage data will missed
        # So, just use all of them...
        if "cpuUsage" in self.dataDF.columns:
            self.dataDF = self.dataDF.loc[
                (self.dataDF["cpuUsage"] > self.cpu_usage_limits["lower"])
                & (self.dataDF["cpuUsage"] < self.cpu_usage_limits["higher"])
            ]
            accepted_ms = (
                self.dataDF.groupby(["service", "microservice"])
                .apply(
                    lambda x: x.groupby(
                        ["repeat", "service", "cpuInter", "memInter", "targetReqFreq"]
                    )["cpuUsage"]
                    .mean()
                    .quantile(0.90)
                    .max()
                    >= self.min_of_max_cpu
                )
                .reset_index()
            )
            accepted_ms = accepted_ms.loc[accepted_ms[0]]
            accepted_ms = [
                (x[1]["service"], x[1]["microservice"])
                for x in accepted_ms.to_dict(orient="index").items()
            ]
            all_ms = (
                self.dataDF.apply(lambda x: (x["service"], x["microservice"]), axis=1)
                .drop_duplicates()
                .to_list()
            )
            other_ms = [x for x in all_ms if x not in accepted_ms]
            other_data_df = self.dataDF.loc[
                self.dataDF.apply(
                    lambda x: (x["service"], x["microservice"]) in other_ms, axis=1
                )
            ]
            self.low_usage_ms_df = other_data_df
            self.dataDF = self.dataDF.loc[
                self.dataDF.apply(
                    lambda x: (x["service"], x["microservice"]) in accepted_ms, axis=1
                )
            ]
        gap = self.dataDF["reqFreq"] - self.dataDF["targetReqFreq"]
        error = abs(gap / self.dataDF["targetReqFreq"])
        acceptable_error = self.throughput_error_rate
        self.dataDF = self.dataDF.loc[error < acceptable_error]
        # Classify reqFreq
        self.dataDF = (
            self.dataDF.groupby("service")
            .apply(
                lambda x: (
                    x.drop(columns="reqFreq").assign(
                        reqFreq=(
                            (
                                x["reqFreq"] / self.class_gap[x["service"].iloc[0]]
                            ).astype(int)
                            * self.class_gap[x["service"].iloc[0]]
                        )
                    )
                )
            )
            .drop(columns="service")
            .reset_index()
        )
        # Remove data that don't have enough repeat
        repeats_count = (
            self.dataDF.groupby(
                ["service", "microservice", "cpuInter", "memInter", "targetReqFreq"]
            )["repeat"]
            .nunique()
            .reset_index()
        )
        # repeats_count = repeats_count.loc[repeats_count["repeat"] >= 3].drop(columns="repeat")
        # keys = list(repeats_count.columns.values)
        # i1 = self.dataDF.set_index(keys).index
        # i2 = repeats_count.set_index(keys).index
        # self.dataDF = pd.concat([
        #     self.dataDF[i1.isin(i2)].assign(enough=True),
        #     self.dataDF[~i1.isin(i2)].assign(enough=False)
        # ])
        # Calculate the replicas
        replicaDF = (
            self.dataDF.groupby(["service", "microservice"])
            .apply(
                lambda x: replica_dict[x["service"].iloc[0]][x["microservice"].iloc[0]]
                if x["service"].iloc[0] in replica_dict
                and x["microservice"].iloc[0] in replica_dict[x["service"].iloc[0]]
                else 1
            )
            .reset_index()
            .rename(columns={0: "replicas"})
        )
        # Take mean of latency of each pod
        self.dataDF = (
            self.dataDF.groupby(
                ["service", "microservice", "cpuInter", "memInter", "reqFreq"]
            )
            .quantile(0.5)
            .reset_index()
        )
        self.dataDF = self.dataDF.merge(replicaDF, on=["service", "microservice"])
        self.dataDF["reqFreq"] /= self.dataDF["replicas"]
        # tagging data
        service = self.dataDF["service"].astype(str)
        cpu = self.dataDF["cpuInter"].astype(str)
        mem = self.dataDF["memInter"].astype(str)
        ms = self.dataDF["microservice"].astype(str)
        self.dataDF.insert(0, "tag", service + "_" + cpu + "_" + mem + "_" + ms)
        # sorting data
        self.dataDF = self.dataDF.sort_values("reqFreq")
        # smoothing data
        # self.dataDF = self.dataDF.groupby([
        #     "service", "microservice", "cpuInter", "memInter"
        # ]).apply(lambda x: self.dataSmoothing(x, "latency"))
        # self.dataDF = (
        #     self.dataDF
        #     .drop(columns=["service", "microservice", "cpuInter", "memInter"])
        #     .reset_index()
        # )
        # Remove useless column
        self.dataDF = self.dataDF[
            [
                "tag",
                "service",
                "microservice",
                "cpuInter",
                "memInter",
                "reqFreq",
                "latency",
                "replicas",
                "cpuUsage",
                "memUsage",
            ]
        ]
        if self.draw_figures:
            self._draw_original_fig()

    def _draw_original_fig(self):
        os.system(f"mkdir -p {self.figure_path}/original")
        from utils.figurePainter import draw_figure

        data = self.dataDF
        for service in data["service"].unique().tolist():
            service_data = data.loc[data["service"] == service]
            for ms in service_data["microservice"].unique().tolist():
                ms_data = service_data.loc[service_data["microservice"] == ms]
                figure_data = []
                for tag in ms_data["tag"].unique().tolist():
                    tag_data = ms_data.loc[ms_data["tag"] == tag]
                    _, cpu, mem, _ = str(tag).split("_")
                    drawing_data = {
                        "draws": [
                            {
                                "data": (
                                    tag_data[["reqFreq", "latency"]].rename(
                                        columns={"reqFreq": "x", "latency": "y"}
                                    )
                                ),
                                "type": "scatter",
                                "color": "b",
                            }
                        ],
                        "title": f"{cpu} cores, {mem}MB",
                    }
                    figure_data.append(drawing_data)
                draw_figure(
                    figure_data, f"{self.figure_path}/original/{service}_{ms}.png"
                )

    def _draw_erms_fitting_fig(self, model_data: pd.DataFrame, figure_type):
        from utils.figurePainter import draw_figure

        for service in model_data["service"].unique().tolist():
            service_models = model_data.loc[model_data["service"] == service]
            for ms in service_models["microservice"].unique().tolist():
                data = self.dataDF.loc[
                    self.dataDF["tag"].str.contains(rf"{service}_.*_{ms}")
                ]
                figure_data = []
                for tag, group_data in data.groupby("tag"):
                    (_, cpu, mem, _) = str(tag).split("_")
                    draws_data = []
                    # Draw original data
                    draws_data.append(
                        {
                            "data": (
                                group_data[["reqFreq", "latency"]].rename(
                                    columns={"reqFreq": "x", "latency": "y"}
                                )
                            ),
                            "type": "scatter",
                            "color": "b",
                        }
                    )
                    ms_models = service_models.loc[service_models["microservice"] == ms]
                    # Get inflection point, draw it
                    inflection_data = self.inflection_points.loc[
                        (self.inflection_points["tag"] == f"{service}_{ms}")
                        & (self.inflection_points["cpuInter"] == cpu)
                        & (self.inflection_points["memInter"] == mem)
                    ].iloc[0]
                    if not inflection_data["success"]:
                        figure_data.append(
                            {"draws": draws_data, "title": f"{cpu} cores, {mem}MB"}
                        )
                        continue
                    inflection = inflection_data["inflection"]
                    draws_data.append(
                        {"data": inflection, "type": "axvline", "color": "g"}
                    )
                    # Generate diagram before inflection & after inflection
                    for i in range(2):
                        model_type, part_data = [
                            (
                                "beforeInflection",
                                group_data.loc[group_data["reqFreq"] <= inflection],
                            ),
                            (
                                "afterInflection",
                                group_data.loc[group_data["reqFreq"] >= inflection],
                            ),
                        ][i]
                        if len(part_data) < 2 or len(ms_models) < 2:
                            continue
                        model_path = ms_models[ms_models["type"] == model_type].iloc[0][
                            "modelFile"
                        ]
                        with open(model_path, "rb") as file:
                            model = pickle.load(file)
                        x = part_data["reqFreq"]
                        y = model.predict(
                            x, part_data["cpuInter"], part_data["memInter"]
                        )
                        draws_data.append(
                            {
                                "data": pd.DataFrame(
                                    list(zip(x, y)), columns=["x", "y"]
                                ),
                                "type": "line",
                                "color": "r",
                            }
                        )
                    figure_data.append(
                        {"draws": draws_data, "title": f"{cpu} cores, {mem}MB"}
                    )
                draw_figure(
                    figure_data, f"{self.figure_path}/{figure_type}/{service}_{ms}.png"
                )

    def _drawInflectionFigure(
        self,
        data_df: pd.DataFrame,
        inflection_data: pd.DataFrame,
        figure_type="inflection",
    ):
        if figure_type == "inflection":
            os.system(f"mkdir -p {self.figure_path}/inflection")
        elif figure_type == "intermediate":
            os.system(f"mkdir -p {self.figure_path}/intermediate")
        serviceList = data_df["service"].unique().tolist()
        for service in serviceList:
            for ms in (
                data_df.loc[data_df["service"] == service]["microservice"]
                .unique()
                .tolist()
            ):
                plt.clf()
                data = data_df.loc[
                    (data_df["service"] == service) & (data_df["microservice"] == ms)
                ]
                figSize = len(data["tag"].unique().tolist())
                edge = ceil(sqrt(figSize))
                (_, axs) = plt.subplots(
                    edge, edge, figsize=(edge * 6, edge * 6), squeeze=False
                )
                plt.suptitle(f"{service}_{ms}")
                for index, group in enumerate(data.groupby(["tag"])):
                    (_, cpu, mem, _) = group[0].split("_")
                    row = floor(index / edge)
                    col = index % edge
                    # prepare subplot
                    subplot = axs[row, col]
                    subplot.set_title(f"{cpu}_{mem}")
                    # draw original data
                    x = group[1]["reqFreq"]
                    y = group[1]["latency"]
                    subplot.scatter(x, y, color="b")
                    single_inflection = inflection_data.loc[
                        (inflection_data["tag"] == f"{service}_{ms}")
                        & (inflection_data["cpuInter"] == cpu)
                        & (inflection_data["memInter"] == mem)
                    ].iloc[0]
                    if not single_inflection["success"]:
                        continue
                    # draw part1 fitting
                    x1 = x.loc[x <= single_inflection["inflection"]]
                    y1 = linear(x1, single_inflection["a1"], single_inflection["b1"])
                    subplot.plot(x1, y1, color="r")
                    # draw part2 fitting
                    x2 = x.loc[x > single_inflection["inflection"]]
                    y2 = linear(x2, single_inflection["a2"], single_inflection["b2"])
                    subplot.plot(x2, y2, color="r")
                    # Draw inflection
                    subplot.axvline(
                        x=single_inflection["inflection"], linestyle="--", color="g"
                    )
                if figure_type == "inflection":
                    plt.savefig(f"{self.figure_path}/inflection/{service}_{ms}.png")
                elif figure_type == "intermediate":
                    plt.savefig(f"{self.figure_path}/intermediate/{service}_{ms}.png")
