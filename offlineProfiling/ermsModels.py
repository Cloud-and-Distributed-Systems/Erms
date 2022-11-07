import pickle
from scipy.optimize import curve_fit


def erms_model_slope(X, a, b, c):
    cpu, memory = X
    return a * cpu + b * memory + c


def erms_model_full(X, a, b, c, d):
    throughput, cpu, memory = X
    return (a * cpu + b * memory + c) * throughput + d


def usage_linear(throughput, a, b):
    return a * throughput + b


def to_float(data):
    try:
        return float(data)
    except:
        return data.astype(float)


class ErmsModel:
    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def dump(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def predict(self, throughput, cpu_inter, mem_inter):
        pass

    def predict_slope(self, cpu_inter, mem_inter):
        pass

    def get_bias(self):
        return self.d

    @staticmethod
    def load(file_path) -> "ErmsModel":
        with open(file_path, "rb") as file:
            return pickle.load(file)


class FullFitting(ErmsModel):
    def __init__(self, a=0, b=0, c=0, d=0):
        super().__init__(a, b, c, d)

    def train(self, throughput, cpu_inter, mem_inter, latency):
        throughput = to_float(throughput)
        cpu_inter = to_float(cpu_inter)
        mem_inter = to_float(mem_inter)
        latency = to_float(latency)

        popt, _ = curve_fit(
            erms_model_full, (throughput, cpu_inter, mem_inter), latency
        )
        self.a = popt[0]
        self.b = popt[1]
        self.c = popt[2]
        self.d = popt[3]

    def predict(self, throughput, cpu_inter, mem_inter):
        throughput = to_float(throughput)
        cpu_inter = to_float(cpu_inter)
        mem_inter = to_float(mem_inter)
        return erms_model_full(
            (throughput, cpu_inter, mem_inter), self.a, self.b, self.c, self.d
        )

    def predict_slope(self, cpu_inter, mem_inter):
        cpu_inter = to_float(cpu_inter)
        mem_inter = to_float(mem_inter)
        return erms_model_slope((cpu_inter, mem_inter), self.a, self.b, self.c)

    def score(self, throughput, cpu_inter, mem_inter, latency):
        prediction = self.predict(throughput, cpu_inter, mem_inter)
        accy = 1 - abs(prediction - latency) / latency
        if not isinstance(accy, float):
            accy = accy.quantile(0.5)
        return accy


class FittingBySlope(ErmsModel):
    def __init__(self, a=0, b=0, c=0, d=0):
        super().__init__(a, b, c, d)

    def train(self, cpu_inter, mem_inter, slope, bias):
        cpu_inter = to_float(cpu_inter)
        mem_inter = to_float(mem_inter)
        slope = to_float(slope)
        bias = to_float(bias)

        popt, _ = curve_fit(erms_model_slope, (cpu_inter, mem_inter), slope)
        self.a = popt[0]
        self.b = popt[1]
        self.c = popt[2]
        if isinstance(bias, float):
            self.d = bias
        else:
            self.d = bias.quantile(0.5)

    def predict_slope(self, cpu_inter, mem_inter):
        cpu_inter = to_float(cpu_inter)
        mem_inter = to_float(mem_inter)
        return erms_model_slope((cpu_inter, mem_inter), self.a, self.b, self.c)

    def predict(self, throughput, cpu_inter, mem_inter):
        return self.predict_slope(cpu_inter, mem_inter) * throughput + self.d

    def score(self, cpu_inter, mem_inter, slope):
        prediction = self.predict_slope(cpu_inter, mem_inter)
        accy = 1 - abs(prediction - slope) / slope
        if not isinstance(accy, float):
            accy = accy.quantile(0.5)
        return accy


class FittingUsage:
    def __init__(self):
        self.a = 0
        self.b = 0

    def train(self, throughput, usage):
        throughput = to_float(throughput)
        usage = to_float(usage)
        popt, _ = curve_fit(usage_linear, throughput, usage)
        self.a = popt[0]
        self.b = popt[1]

    def predict(self, throughput):
        throughput = to_float(throughput)
        return usage_linear(throughput, self.a, self.b)

    def score(self, throughput, usage):
        throughput = to_float(throughput)
        usage = to_float(usage)
        prediction = self.predict(throughput)
        accy = 1 - abs(prediction - usage) / usage
        if not isinstance(accy, float):
            accy = accy.quantile(0.5)
        return accy
