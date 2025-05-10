import sys
import pandas as pd
from typing import List, Optional, Tuple
import math
import numpy as np
from skimage.restoration import denoise_wavelet
from src.exception import CustomException


def discrete_wavelet_transform(
    df: pd.DataFrame, column: str = "Close", inplace: bool = True
) -> Optional[pd.DataFrame] | np.ndarray:
    """
    Performs noise smoothing on the given data using the Haar wavelet transform.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str, optional): Column name to denoise. Defaults to "Close".
        inplace (bool, optional): Whether to alter the input DataFrame or return the transformed signal only. Defaults to True.

    Returns:
        Optional[pd.DataFrame] | pd.Series: Output DataFrame or Series depending on the value of inplace.
    """
    try:
        smoothed = denoise_wavelet(df[column], wavelet="haar", mode="soft")
        if inplace:
            df[column] = smoothed
            return None
        return smoothed
    except Exception as e:
        raise CustomException(e, sys)


def train_val_test_ordered_split(
    df: pd.DataFrame, train_split: float = 0.64, val_split: float = 0.16
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orders the data and organizes it into training, validation and testing dataframes.

    Args:
        df (pd.DataFrame): Input DataFrame.
        train_split (float, optional): Fraction of data to use for training. Defaults to 0.64.
        val_split (float, optional): Fraction of data to use for validation. Defaults to 0.16.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, val and test dataframes.
    """
    try:
        df_copy = df.sort_index()
        train_inds = (0, math.ceil(len(df_copy) * train_split))
        val_inds = (train_inds[1], train_inds[1] + math.ceil(len(df_copy) * val_split))
        test_inds = (val_inds[1], len(df_copy))
        return (
            df_copy.iloc[: train_inds[1]],
            df_copy.iloc[val_inds[0] : val_inds[1]],
            df_copy.iloc[test_inds[0] :],
        )
    except Exception as e:
        raise CustomException(e, sys)


def lookback_format(
    df: pd.DataFrame,
    window: int = 15,
    target_column: str = "Close",
    classification: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the lookback format to the data based on the window size W given.
    Given an input dataframe of shape (N, F) where N is the number of rows and F is the number of  features,
    the output X will be of shape (N, W, F) and the output y will be a vector of size N.

    Args:
        df (pd.DataFrame): Input DataFrame.
        window (int, optional): Size of the lookback window. Defaults to 30.
        target_column (str, optional): Name of the target column. Defaults to "Close".
        classification (bool, optional): Whether the target column is a classification or regression. Defaults to True.

    Returns:
        (X, y): X is a 3D-array of shape (N, W, F) and y is a 1D-array of size N.
    """
    try:
        dataset_np = df.to_numpy()
        X = []
        for i in range(window, len(dataset_np)):
            X.append(dataset_np[i - window : i])

        y = df[target_column].iloc[window - 1 :].to_numpy()

        if classification:
            y = np.array([1 if y[i] > y[i - 1] else 0 for i in range(1, len(y))])
            y = y.astype(int)
        else:
            y = y[1:]

        return np.array(X), y
    except Exception as e:
        raise CustomException(e, sys)


def get_trend(df: pd.DataFrame, column: str = "Close", first: float = 0.0) -> pd.Series:
    """
    Computes the trend of the given column in the DataFrame.
    """
    try:
        trend = np.zeros(len(df), dtype=np.float32)
        if len(df) == 1:
            return pd.Series(trend, index=df.index, name="Trend")
        for i in range(len(df) - 1):
            if df[column].iloc[i] <= df[column].iloc[i + 1]:
                trend[i] = 1.0
        return pd.Series(trend, index=df.index, name="Trend")
    except Exception as e:
        raise CustomException(e, sys)
