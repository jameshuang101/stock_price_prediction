import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from src.logger import logging
from typing import List, Optional, Tuple
from src.exception import CustomException
from src.utils import save_object, load_object
import dateparser
from src.preprocessing import (
    data_cleaning,
    data_ingestion,
    data_transformation,
    indicators,
)
import sys
import pandas as pd
import joblib
import yaml
import os
import pickle
from os.path import join, dirname, abspath
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from datetime import datetime, date, timedelta
from pathlib import Path

API_KEYS_PATH = join(
    Path(__file__).parent.parent.parent.parent.resolve(), "api_keys.yml"
)
try:
    with open(API_KEYS_PATH, "r") as f:
        FRED_API_KEY = yaml.safe_load(f)["fred_api_key"]
except Exception as e:
    logging.info(f"Failed to load FRED API key: {e}")
    raise CustomException(e, sys)


class StockDataset(Dataset):
    def __init__(
        self,
        stock: Optional[str] = None,
        dict_path: Optional[str] = None,
        scaler: Optional[RobustScaler | StandardScaler | MinMaxScaler] = None,
        date: Optional[str | datetime] = None,
        start_date: Optional[str | datetime] = None,
        end_date: Optional[str | datetime] = None,
        targets: Optional[List[str]] = ["trend", "peaks", "valleys"],
    ):
        if dict_path is not None:
            logging.info("Loading data dict from file...")
            try:
                data_dict = load_object(file_path=dict_path)
                self._stock = data_dict["stock"]
                self._data = data_dict["data"]
                self._scaler = data_dict["scaler"]
                self.X = self._scaler.transform(
                    self._data.drop(
                        columns=["Open", "Close", "High", "Low", "Volume"]
                    ).to_numpy(dtype=np.float32)
                )
                targets_dict = dict()
                targets_dict["trend"] = data_transformation.get_trend(
                    self._data
                ).to_numpy(dtype=np.float32)
                targets_dict["peaks"] = data_transformation.get_peaks(
                    self._data
                ).to_numpy(dtype=np.float32)
                targets_dict["valleys"] = data_transformation.get_valleys(
                    self._data
                ).to_numpy(dtype=np.float32)
                self.y = np.stack(
                    [targets_dict[target] for target in targets],
                    axis=1,
                )
            except Exception as e:
                logging.info(f"Failed to load data dict: {e}")
                raise CustomException(e, sys)
            return

        if stock is None:
            logging.info("No stock name provided, aborting prediction")
            raise CustomException("No stock name provided", sys)
        self._stock = stock

        if date is not None:
            start_date = date
            end_date = date

        if start_date is None or end_date is None:
            logging.info("No date provided, aborting prediction")
            raise CustomException("No date(s) provided", sys)

        if not data_ingestion.is_market_day(start_date=start_date, end_date=end_date):
            logging.info("No market days between provided dates, aborting prediction")
            raise CustomException("No market days between provided dates", sys)

        if type(start_date) == str:
            try:
                start_date = dateparser.parse(start_date)
            except:
                logging.info("Invalid date format, aborting prediction")
                raise CustomException("Invalid date format", sys)

        if type(end_date) == str:
            try:
                end_date = dateparser.parse(end_date)
            except:
                logging.info("Invalid date format, aborting prediction")
                raise CustomException("Invalid date format", sys)

        end_date = end_date + timedelta(days=1)

        logging.info(
            f"Grabbing lead market dates for financial indicator calculations..."
        )
        try:
            lead_date = data_ingestion.get_lead_days(date=start_date)[0]
        except Exception as e:
            logging.info(f"Failed to grab lead dates: {e}")
            raise CustomException(e, sys)
        logging.info(f"Success! Lead date is {lead_date}")

        logging.info(
            f"Grabbing stock data for {stock} from {lead_date} to {end_date}..."
        )
        try:
            stock_data = data_ingestion.get_stock_data(self._stock, lead_date, end_date)
        except Exception as e:
            logging.info(f"Failed to grab stock data: {e}")
            raise CustomException(e, sys)
        logging.info(f"Success! Grabbed {len(stock_data)} rows of data.")

        logging.info("Calculating financial indicators and trend...")
        try:
            self._data = indicators.get_technical_indicators(stock_data, inplace=False)
            self.trend = data_transformation.get_trend(self._data)
        except Exception as e:
            logging.info(f"Failed to calculate financial indicators and trend: {e}")
            raise CustomException(e, sys)
        logging.info(f"Successfully calculated financial indicators and trend.")

        logging.info("Formatting and cleaning data...")
        try:
            self._data = data_cleaning.remove_inf_and_nan(self._data, behavior="impute")
            self._data = self._data.loc[start_date:]
            targets_dict = dict()
            targets_dict["trend"] = data_transformation.get_trend(self._data).to_numpy(
                dtype=np.float32
            )
            targets_dict["peaks"] = data_transformation.get_peaks(self._data).to_numpy(
                dtype=np.float32
            )
            targets_dict["valleys"] = data_transformation.get_valleys(
                self._data
            ).to_numpy(dtype=np.float32)
        except Exception as e:
            logging.info(f"Failed to format and clean data: {e}")
            raise CustomException(e, sys)
        logging.info(f"Successfully formatted and cleaned data.")

        logging.info(f"Grabbing macroeconomic data from {lead_date} to {end_date}...")
        try:
            macro_data = data_ingestion.get_macro_data(
                fred_api_key=FRED_API_KEY,
                start_date=lead_date,
                end_date=end_date,
            )
            self._data = self._data.merge(
                macro_data, how="left", left_index=True, right_index=True
            )
        except Exception as e:
            logging.info(f"Failed to grab macroeconomic data: {e}")
            raise CustomException(e, sys)
        logging.info(f"Success! Grabbed and merged {len(macro_data)} rows of data.")

        logging.info("Scaling data...")
        try:
            X = self._data.drop(columns=["Open", "Close", "High", "Low"]).to_numpy(
                dtype=np.float32
            )
            if scaler is not None:
                self._scaler = scaler
            else:
                self._scaler = MinMaxScaler(feature_range=(0, 1))
                self._scaler.fit(X)
            self.X = self._scaler.transform(X)
            self.y = np.stack(
                [targets_dict[target] for target in targets],
                axis=1,
            )
        except Exception as e:
            logging.info(f"Failed to scale data: {e}")
            raise CustomException(e, sys)
        logging.info(f"Successfully scaled data.")
        logging.info("Successfully initialized dataset.")

    def save_data_dict(self, dict_path: str):
        try:
            data_dict = {
                "data": self._data,
                "scaler": self._scaler,
                "stock": self._stock,
            }
            save_object(file_path=dict_path, obj=data_dict)
        except Exception as e:
            raise CustomException(f"Failed to save data dict: {e}", sys)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def data(self):
        return self._data

    @property
    def scaler(self):
        return self._scaler

    @property
    def stock(self):
        return self._stock
