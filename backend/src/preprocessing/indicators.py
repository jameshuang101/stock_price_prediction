import sys
import pandas as pd
from typing import List, Optional, Tuple
import math
import numpy as np
from skimage.restoration import denoise_wavelet
from src.exception import CustomException


def log_price(
    data: pd.DataFrame,
    col1: str = "Close",
    col2: str = "Close",
    shift1: int = 0,
    shift2: int = 1,
) -> pd.Series:
    """
    Calculates the natural log of col1 at shift1 over col2 at shift2 over the input data.
    For example, given the default arguments, the function will calculate the ln of Close over the Close of the previous day.
    """
    try:
        numerator = data[col1].shift(shift1)
        denominator = data[col2].shift(shift2)
        return np.log(numerator / denominator)
    except Exception as e:
        raise CustomException(e, sys)


def ema(data: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """
    Calculates the exponential moving average (EMA) for given period over the input data.
    """
    try:
        return data[column].ewm(span=period, adjust=False).mean()
    except Exception as e:
        raise CustomException(e, sys)


def sma(data: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """
    Calculates the simple moving average (SMA) for given period over the input data.
    """
    try:
        return data[column].rolling(window=period).mean()
    except Exception as e:
        raise CustomException(e, sys)


def wma(data: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """
    Calculates the weighted moving average (WMA) for given period over the input data.
    """
    try:
        return (
            data[column]
            .rolling(window=period)
            .apply(
                lambda x: np.sum(x * np.arange(1, period + 1))
                / np.sum(np.arange(1, period + 1)),
                raw=True,
            )
        )
    except Exception as e:
        raise CustomException(e, sys)


def hma(data: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """
    Calculates the Hull moving average (HMA) for given period over the input data.
    """
    try:
        return (
            2
            * (wma(data, period // 2 + 1, column) - wma(data, period, column))
            .rolling(int(np.sqrt(period)))
            .mean()
        )
    except Exception as e:
        raise CustomException(e, sys)


def stochastic_oscillator(
    data: pd.DataFrame, period: int, column: str = "Close"
) -> pd.DataFrame:
    """
    Calculates the stochastic oscillator for given period over the input data.
    """
    try:
        df = data.copy()
        for i in range(len(data)):
            low = df.iloc[i][column]
            high = df.iloc[i][column]
            if i >= period:
                n = 0
                while n < period:
                    if df.iloc[i - n][column] >= high:
                        high = df.iloc[i - n][column]
                    elif df.iloc[i - n][column] < low:
                        low = df.iloc[i - n][column]
                    n += 1
                df.at[i, "best_low"] = low
                df.at[i, "best_high"] = high
                df.at[i, "Fast_K"] = 100 * (
                    (df.iloc[i][column] - df.iloc[i]["best_low"])
                    / (df.iloc[i]["best_high"] - df.iloc[i]["best_low"])
                )

        data["Fast_K"] = df["Fast_K"]
        data["Fast_D"] = df["Fast_K"].rolling(3).mean().round(2)
        data["Slow_D"] = data["Fast_D"].rolling(3).mean().round(2)

        return data
    except Exception as e:
        raise CustomException(e, sys)


def williams_r(
    data: pd.DataFrame,
    period: int,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    """
    Calculates the Williams %R for given period over the input data.
    """
    try:
        highh = data[high_low_close_cols[0]].rolling(window=period).max()
        lowl = data[high_low_close_cols[1]].rolling(window=period).min()
        return -100 * ((highh - data[high_low_close_cols[2]]) / (highh - lowl))
    except Exception as e:
        raise CustomException(e, sys)


def cci(
    data: pd.DataFrame,
    period: int = 40,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    """
    Calculates the commodity channel index (CCI) for given period over the input data.
    """
    try:
        TP = (
            data[high_low_close_cols[0]]
            + data[high_low_close_cols[1]]
            + data[high_low_close_cols[2]]
        ) / 3
        SMA = sma(data, period=period, column=high_low_close_cols[2])
        mad = TP.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        return (TP - SMA) / (0.015 * mad)
    except Exception as e:
        raise CustomException(e, sys)


def momentum(data: pd.DataFrame, shift: int = 1, column: str = "Close") -> pd.Series:
    """
    Calculates the momentum over the input data.
    """
    try:
        return data[column].diff(periods=shift)
    except Exception as e:
        raise CustomException(e, sys)


def macd(data: pd.DataFrame, column: str = "Close") -> pd.Series:
    """
    Calculates the moving average convergence divergence (MACD) over the input data using
    the formula MACD = EMA26 - EMA12.
    """
    try:
        return ema(data, period=26, column=column) - ema(data, period=12, column=column)
    except Exception as e:
        raise CustomException(e, sys)


def rma(s: pd.Series, period: int) -> pd.Series:
    try:
        return s.ewm(alpha=1 / period).mean()
    except Exception as e:
        raise CustomException(e, sys)


def tr(
    data: pd.DataFrame,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    """
    Calculates the true range (TR) over the input data.
    """
    try:
        high, low, prev_close = (
            data[high_low_close_cols[0]],
            data[high_low_close_cols[1]],
            data[high_low_close_cols[2]].shift(),
        )
        tr_all = [high - low, high - prev_close, low - prev_close]
        tr_all = [tr.abs() for tr in tr_all]
        tr = pd.concat(tr_all, axis=1).max(axis=1)
        return tr
    except Exception as e:
        raise CustomException(e, sys)


def atr(
    data: pd.DataFrame,
    period: int = 14,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    """
    Calculates the average true range (ATR) over the input data for a particular period of time.
    """
    try:
        return rma(tr(data, high_low_close_cols), period)
    except Exception as e:
        raise CustomException(e, sys)


def rsi(data: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """
    Calculates the relative strength index (RSI) over the input data for a particular period of time.
    """
    try:
        change = data[column].diff()
        change.dropna(inplace=True)
        change_up = change.copy()
        change_down = change.copy()
        change_up[change_up < 0] = 0
        change_down[change_down > 0] = 0
        # assert change.equals(change_up+change_down)
        avg_up = change_up.rolling(period, min_periods=1).mean()
        avg_down = change_down.rolling(period, min_periods=1).mean()
        return 100 * avg_up / (avg_up + avg_down)
    except Exception as e:
        raise CustomException(e, sys)


def di(
    data: pd.DataFrame,
    period: int = 14,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    """
    Calculates the directional index (DI) over the input data for a particular period of time.
    """
    try:
        di_plus = 100 * (
            data[high_low_close_cols[0]].diff().ewm(span=period, adjust=False).mean()
            / atr(data, period=period, high_low_close_cols=high_low_close_cols)
        )
        di_minus = 100 * (
            data[high_low_close_cols[1]]
            .diff(periods=-1)
            .shift(1)
            .ewm(span=period, adjust=False)
            .mean()
        )
        return 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).abs()
    except Exception as e:
        raise CustomException(e, sys)


def adx(
    data: pd.DataFrame,
    period: int = 14,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    """
    Calculates the average directional index (ADX) over the input data for a particular period of time.
    """
    try:
        di = di(data, period=period, high_low_close_cols=high_low_close_cols)
        return ((di.shift(1) * 13) + di) / period
    except Exception as e:
        raise CustomException(e, sys)


def psar(
    data: pd.DataFrame,
    af: float = 0.02,
    max: float = 0.2,
    high_low_close_cols: Tuple[str, str, str] = ("High", "Low", "Close"),
) -> pd.Series:
    psar = data.to_dict("index")

    psar[0]["AF"] = af
    psar[0]["PSAR"] = psar[0][high_low_close_cols[1]]
    psar[0]["EP"] = psar[0][high_low_close_cols[0]]
    psar[0]["PSARdir"] = "bull"

    i = list(psar.keys())[1:]

    for i in list(psar.keys())[1:]:  # start on second data row
        prev_i = i - 1
        if psar[prev_i]["PSARdir"] == "bull":
            psar[i]["PSAR"] = psar[prev_i]["PSAR"] + (
                psar[prev_i]["AF"] * (psar[prev_i]["EP"] - psar[prev_i]["PSAR"])
            )
            psar[i]["PSARdir"] = "bull"

            if (
                psar[i]["Low"] < psar[prev_i]["PSAR"]
                or psar[i]["Low"] < psar[i]["PSAR"]
            ):
                psar[i]["PSARdir"] = "bear"
                psar[i]["PSAR"] = psar[prev_i]["EP"]
                psar[i]["EP"] = psar[prev_i]["Low"]
                psar[i]["AF"] = af

            else:
                if psar[i][high_low_close_cols[0]] > psar[prev_i]["EP"]:
                    psar[i]["EP"] = psar[i][high_low_close_cols[0]]
                    psar[i]["AF"] = min(max, psar[prev_i]["AF"] + af)
                else:
                    psar[i]["AF"] = psar[prev_i]["AF"]
                    psar[i]["EP"] = psar[prev_i]["EP"]

        else:
            psar[i]["PSAR"] = psar[prev_i]["PSAR"] - (
                psar[prev_i]["AF"] * (psar[prev_i]["PSAR"] - psar[prev_i]["EP"])
            )
            psar[i]["PSARdir"] = "bear"

            if (
                psar[i][high_low_close_cols[0]] > psar[prev_i]["PSAR"]
                or psar[i][high_low_close_cols[0]] > psar[i]["PSAR"]
            ):
                psar[i]["PSARdir"] = "bull"
                psar[i]["PSAR"] = psar[prev_i]["EP"]
                psar[i]["EP"] = psar[prev_i][high_low_close_cols[0]]
                psar[i]["AF"] = af
            else:
                if psar[i][high_low_close_cols[1]] < psar[prev_i]["EP"]:
                    psar[i]["EP"] = psar[i][high_low_close_cols[1]]
                    psar[i]["AF"] = min(max, psar[prev_i]["AF"] + af)
                else:
                    psar[i]["AF"] = psar[prev_i]["AF"]
                    psar[i]["EP"] = psar[prev_i]["EP"]

    df = pd.DataFrame.from_dict(psar, orient="index")
    return df["PSAR"]


def overnight_percent_diff(
    data: pd.DataFrame, open_close_cols: Tuple[str, str] = ("Open", "Close")
) -> pd.Series:
    """
    Calculates the percent change overnight during market closed hours.
    """
    try:
        return 100 * (
            (data[open_close_cols[0]] - data[open_close_cols[1]].shift(1))
            / data[open_close_cols[1]].shift(1)
        )
    except Exception as e:
        raise CustomException(e, sys)


def get_technical_indicators(
    df: pd.DataFrame,
    high_low_close_open_cols: Tuple[str, str, str, str] = (
        "High",
        "Low",
        "Close",
        "Open",
    ),
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
    try:
        df_copy = df.copy()
        df_copy["MACD"] = macd(df_copy, high_low_close_open_cols[2])
        df_copy["ATR"] = atr(df_copy, high_low_close_cols=high_low_close_open_cols[:-1])
        df_copy["RSI"] = rsi(df_copy, column=high_low_close_open_cols[2])
        return df_copy
    except Exception as e:
        raise CustomException(e, sys)
