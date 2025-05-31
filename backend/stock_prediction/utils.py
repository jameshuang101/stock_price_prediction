import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import torch
from stock_prediction.exception import CustomException
import jsonpickle
import json


def save_object(file_path: str, obj, as_json: bool = False):
    """
    Saves an object to a pickle or JSON file.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        if not as_json:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            json_obj = jsonpickle.encode(obj)
            with open(file_path, "w") as file:
                json.dump(json_obj, file, indent=4)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads an object from a pickle or JSON file.
    """
    try:
        file_ext = os.path.splitext(file_path)[-1]
        if file_ext == ".pickle" or file_ext == ".pkl":
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)
        elif file_ext == ".json":
            with open(file_path, "r") as file_obj:
                return jsonpickle.decode(json.load(file_obj))
        else:
            raise ValueError("Invalid file extension. Must be .pickle or .json")

    except Exception as e:
        raise CustomException(e, sys)


def save_model(epochs, model, optimizer, criterion, path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving model...")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        path,
    )
