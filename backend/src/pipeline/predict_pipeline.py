import math
import os
import numpy as np
import sys
import pandas as pd
from typing import List, Optional, Tuple


class Predictor:

    def __init__(
        self,
        stock: str,
        model_path: str,
        date: str = "",
        start_date: str = "",
        end_date: str = "",
    ):
        self.stock = stock
        self.model_path = model_path
        self.date = date
        self.start_date = start_date
        self.end_date = end_date

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
