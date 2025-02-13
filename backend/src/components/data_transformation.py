import pandas as pd
from typing import List, Optional, Tuple
import math
import numpy as np
from skimage.restoration import denoise_wavelet


def ema(data: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """
    Calculates the exponential moving average (EMA) for given period over the input data.
    """
    return data[column].ewm(span=period, adjust=False).mean()


def macd(data: pd.DataFrame, column: str = "Close") -> pd.Series:
    """
    Calculates the moving average convergence divergence (MACD) over the input data using
    the formula MACD = EMA26 - EMA12.
    """
    return ema(data, period=26, column=column) - ema(data, period=12, column=column)


def rma(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period).mean()


def atr(
    df: pd.DataFrame,
    length: int = 14,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    """
    Calculates the average true range (ATR) over the input data for a particular period of time.
    """
    high, low, prev_close = (
        df[high_low_close_cols[0]],
        df[high_low_close_cols[1]],
        df[high_low_close_cols[2]].shift(),
    )
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)
    return rma(tr, length)


def rsi(df: pd.DataFrame, length: int = 14, column: str = "Close") -> pd.Series:
    """
    Calculates the relative strength index (RSI) over the input data for a particular period of time.
    """
    change = df[column].diff()
    change.dropna(inplace=True)
    change_up = change.copy()
    change_down = change.copy()
    change_up[change_up < 0] = 0
    change_down[change_down > 0] = 0
    # assert change.equals(change_up+change_down)
    avg_up = change_up.rolling(length, min_periods=1).mean()
    avg_down = change_down.rolling(length, min_periods=1).mean()
    return 100 * avg_up / (avg_up + avg_down)


def get_technical_indicators(
    df: pd.DataFrame,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Calculates three financial technical indicators (MACD, ATR, and RSI) for the given data.

    Args:
        df (pd.DataFrame): Input DataFrame.
        high_low_close_cols (Tuple[str, str, str], optional): Column names for high, low and close in df. Defaults to ("High", "Low", "Close").
        inplace (bool, optional): Whether to alter the input DataFrame or return a new one. Defaults to True (alter the current one).

    Returns:
        None if inplace is True, altered DataFrame with technical indicator columns attached if inplace is False.
    """
    if inplace:
        df["MACD"] = macd(df, high_low_close_cols[2])
        df["ATR"] = atr(df, high_low_close_cols=high_low_close_cols)
        df["RSI"] = rsi(df, column=high_low_close_cols[2])
        return None
    df_copy = df.copy()
    df_copy["MACD"] = macd(df_copy, high_low_close_cols[2])
    df_copy["ATR"] = atr(df_copy, high_low_close_cols=high_low_close_cols)
    df_copy["RSI"] = rsi(df_copy, column=high_low_close_cols[2])
    return df_copy


def overnight_percent_diff(
    df: pd.DataFrame, open_close_cols: Tuple[str, str] = ("Open", "Close")
) -> pd.Series:
    """
    Calculates the percent change overnight during market closed hours.
    """
    return 100 * (
        (df[open_close_cols[0]] - df[open_close_cols[1]].shift(1))
        / df[open_close_cols[1]].shift(1)
    )


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
    smoothed = denoise_wavelet(df[column], wavelet="haar", mode="soft")
    if inplace:
        df[column] = smoothed
        return None
    return smoothed


def train_val_test_ordered_split(
    df: pd.DataFrame, train_split: float = 0.64, val_split: float = 0.16
):
    df_copy = df.copy()
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_copy.dropna(inplace=True)
    df_copy.reset_index(inplace=True, drop=True)
    train_inds = (0, math.ceil(len(df_copy) * train_split))
    val_inds = (train_inds[1], train_inds[1] + math.ceil(len(df_copy) * val_split))
    test_inds = (val_inds[1], len(df_copy))
    return (
        df_copy.iloc[: train_inds[1]],
        df_copy.iloc[val_inds[0] : val_inds[1]],
        df_copy.iloc[test_inds[0] :],
    )
