import sys
import pandas as pd
from typing import List, Optional, Tuple
import math
import numpy as np
from skimage.restoration import denoise_wavelet
from stock_prediction.exception import CustomException
from scipy import signal


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


def get_trend(df: pd.DataFrame, column: str = "Close", days_out: int = 1) -> pd.Series:
    """
    Computes the trend of the given column in the DataFrame.
    """
    try:
        trend = np.sign(df[column].diff(periods=days_out))
        trend[trend < 0] = 0.0
        return pd.Series(trend, index=df.index, name="Trend").shift(-days_out).fillna(0)
    except Exception as e:
        raise CustomException(e, sys)


def get_return(df: pd.DataFrame, column: str = "Close", days_out: int = 1) -> pd.Series:
    """
    Computes the return of the given column in the DataFrame.
    """
    try:
        out = df[column].diff(periods=days_out)
        return pd.Series(out, index=df.index, name="Return").shift(-days_out).fillna(0)
    except Exception as e:
        raise CustomException(e, sys)


def get_peaks(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """
    Computes the peaks of the given column in the DataFrame.
    """
    try:
        peaks = np.zeros(len(df), dtype=np.float32)
        peak_inds, _ = signal.find_peaks(df[column])
        peaks[peak_inds] = 1.0
        return pd.Series(peaks, index=df.index, name="Peaks")
    except Exception as e:
        raise CustomException(e, sys)


def get_valleys(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """
    Computes the valleys of the given column in the DataFrame.
    """
    try:
        valleys = np.zeros(len(df), dtype=np.float32)
        valley_inds, _ = signal.find_peaks(-df[column])
        valleys[valley_inds] = 1.0
        return pd.Series(valleys, index=df.index, name="Valleys")
    except Exception as e:
        raise CustomException(e, sys)


def get_buy_reco(
    df: pd.DataFrame,
    pct_thresholds: List[float] = [-0.005, 0.005],
    column: str = "Close",
) -> pd.Series:
    try:
        pct_change = df[column].pct_change().shift(-1).fillna(0)
        pct_thresholds = (
            [min(pct_change.min() - 0.001, -100)]
            + pct_thresholds
            + [max(pct_change.max() + 0.001, 100)]
        )
        return pd.cut(
            pct_change,
            pct_thresholds,
            labels=False,
            duplicates="drop",
        )
    except Exception as e:
        raise CustomException(e, sys)
