import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.logger import logging
from typing import List, Optional, Tuple
from src.exception import CustomException
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
from sklearn.preprocessing import RobustScaler, StandardScaler

API_KEYS_PATH = join(abspath(dirname(dirname(os.getcwd()))), "api_keys.yml")
try:
    with open(API_KEYS_PATH, "r") as f:
        FRED_API_KEY = yaml.safe_load(f)["fred_api_key"]
except Exception as e:
    logging.info(f"Failed to load FRED API key: {e}")
    raise CustomException(e, sys)


class StockDataset(Dataset):
    def __init__(
        self,
        stock: str,
        dict_path: Optional[str] = None,
        scaler: Optional[RobustScaler | StandardScaler] = None,
        date=None,
        start_date=None,
        end_date=None,
    ):
        self.stock = stock
        if dict_path is not None:
            logging.info("Loading data dict from file...")
            try:
                with open(dict_path, "rb") as f:
                    data_dict = pickle.load(f)
                self.data = data_dict["data"]
                if scaler is None:
                    self.scaler = data_dict["scaler"]
                else:
                    self.scaler = scaler
                self.trend = data_transformation.get_trend(self.data)
                self.X = scaler.transform(
                    self.data.drop(
                        columns=["Open", "Close", "High", "Low", "Volume"]
                    ).to_numpy()
                )
                self.y = self.trend.to_numpy()
            except Exception as e:
                logging.info(f"Failed to load data dict: {e}")
                raise CustomException(e, sys)
            return

        if date is not None:
            start_date = date
            end_date = date

        if start_date is None or end_date is None:
            logging.info("No date provided, aborting prediction")
            raise CustomException("No date(s) provided", sys)

        try:
            start_date = dateparser.parse(start_date)
            end_date = dateparser.parse(end_date)
        except:
            logging.info("Invalid date format, aborting prediction")
            raise CustomException("Invalid date format", sys)

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
            stock_data = data_ingestion.get_stock_data(self.stock, lead_date, end_date)
        except Exception as e:
            logging.info(f"Failed to grab stock data: {e}")
            raise CustomException(e, sys)
        logging.info(f"Success! Grabbed {len(stock_data)} rows of data.")

        logging.info("Calculating financial indicators and trend...")
        try:
            self.data = indicators.get_technical_indicators(stock_data, inplace=False)
            self.trend = data_transformation.get_trend(self.data)
        except Exception as e:
            logging.info(f"Failed to calculate financial indicators and trend: {e}")
            raise CustomException(e, sys)
        logging.info(f"Successfully calculated financial indicators and trend.")

        logging.info("Formatting and cleaning data...")
        try:
            self.data = data_cleaning.remove_inf_and_nan(self.data, behavior="impute")
            self.data = self.data.loc[start_date:]
            self.trend = self.trend.loc[start_date:]
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
            self.data = self.data.merge(
                macro_data, how="left", left_index=True, right_index=True
            )
        except Exception as e:
            logging.info(f"Failed to grab macroeconomic data: {e}")
            raise CustomException(e, sys)
        logging.info(f"Success! Grabbed and merged {len(macro_data)} rows of data.")

        logging.info("Scaling data...")
        try:
            X = self.data.drop(
                columns=["Open", "Close", "High", "Low", "Volume"]
            ).to_numpy()
            self.scaler = RobustScaler()
            self.scaler.fit(X)
            self.X = self.scaler.transform(X)
            self.y = self.trend.to_numpy()
        except Exception as e:
            logging.info(f"Failed to scale data: {e}")
            raise CustomException(e, sys)
        logging.info(f"Successfully scaled data.")
        logging.info("Successfully initialized dataset.")

    def save_data_dict(self, dict_path: str):
        try:
            with open(dict_path, "wb") as f:
                pickle.dump(
                    {
                        "data": self.data,
                        "scaler": self.scaler,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            raise CustomException(f"Failed to save data dict: {e}", sys)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
