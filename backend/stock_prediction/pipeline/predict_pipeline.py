import math
import os
import numpy as np
import sys
import pandas as pd
from typing import List, Optional, Tuple
from stock_prediction.preprocessing import (
    data_ingestion,
    data_transformation,
    indicators,
    data_cleaning,
)
import dateparser
from stock_prediction.logger import logging
from stock_prediction.exception import CustomException
from os.path import join, dirname, abspath
import joblib
import yaml

logging.info(
    "Attempting to load FRED API key from ~/stock_price_prediction/api_keys.yml..."
)
API_KEYS_PATH = join(abspath(dirname(dirname(dirname(os.getcwd())))), "api_keys.yml")
try:
    with open(API_KEYS_PATH, "r") as f:
        FRED_API_KEY = yaml.safe_load(f)["fred_api_key"]
except Exception as e:
    logging.info(f"Failed to load FRED API key: {e}")
    raise CustomException(e, sys)


class Predictor:

    def __init__(
        self,
        stock: str,
        model_path: str,
        date=None,
        start_date=None,
        end_date=None,
    ):
        self.stock = stock
        self.model_path = model_path
        self.date = date
        self.start_date = start_date
        self.end_date = end_date

    def load_model(self, model_path: str) -> bool:
        try:
            self.model = joblib.load(model_path)
        except:
            return False
        return True

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
