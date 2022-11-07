from typing import Union
import os
import pickle
import pandas as pd

from offlineProfiling.ermsModels import ErmsModel, FittingUsage as UsageModel
from sklearn.tree import DecisionTreeRegressor


def append_data(data: pd.DataFrame, data_path):
    """This method is used to write data to a file, if the file is exist,
    it will append data to the end, otherwise it will create a new file.

    Args:
        data (pd.DataFrame): Data to save
        data_path (str): Path to the file
    """
    if not os.path.exists(data_path):
        open(data_path, "w").close()
    is_empty = os.path.getsize(data_path) == 0
    data.to_csv(data_path, index=False, mode="a", header=is_empty)

def dump_model(model, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(model, file)

def load_model(file_path) -> Union[ErmsModel, UsageModel, DecisionTreeRegressor]:
    with open(file_path, "rb") as file:
        return pickle.load(file)
