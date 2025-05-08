import math
import os
import numpy as np
import sys
import pandas as pd
from typing import List, Optional, Tuple
from src.preprocessing import data_ingestion
import dateparser


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

        if date is not None:
            self.start_date = self.date
            self.end_date = self.date

        if self.start_date is None or self.end_date is None:
            raise ValueError("No date(s) provided")

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
