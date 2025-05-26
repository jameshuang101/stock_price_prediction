import sys
import pandas as pd
from typing import List, Optional, Tuple
import math
import numpy as np
from skimage.restoration import denoise_wavelet
from src.exception import CustomException
import time


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


def ema(data: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """
    Calculates the exponential moving average (EMA) for given period over the input data.
    """
    try:
        return data[column].ewm(span=period, adjust=False).mean()
    except Exception as e:
        raise CustomException(e, sys)


def sma(data: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """
    Calculates the simple moving average (SMA) for given period over the input data.
    """
    try:
        return data[column].rolling(window=period).mean()
    except Exception as e:
        raise CustomException(e, sys)


def wma(data: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
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


def hma(data: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
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
    data: pd.DataFrame,
    period: int = 14,
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the stochastic oscillators for given period over the input data according to the formulas:
    Fast_K = 100 * (CURRENT_CLOSE - 14 DAY LOW) / (14 DAY HIGH - 14 DAY LOW)
    FAST_D = Fast_K.rolling(3).mean()
    SLOW_D = FAST_D.rolling(3).mean()
    """
    try:
        df = data.copy()
        Fast_K = np.full(len(df), np.nan)
        for i in range(period, len(df)):
            LOW14 = df.iloc[i - period : i][ohlcv_cols[2]].min()
            HIGH14 = df.iloc[i - period : i][ohlcv_cols[1]].max()
            Fast_K[i] = 100 * (df.iloc[i][ohlcv_cols[3]] - LOW14) / (HIGH14 - LOW14)
        df["Fast_K"] = Fast_K
        df["Fast_D"] = df["Fast_K"].rolling(3).mean()
        df["Slow_D"] = df["Fast_D"].rolling(3).mean()
        if inplace:
            data = df
        return df
    except Exception as e:
        raise CustomException(e, sys)


def williams_r(
    data: pd.DataFrame,
    period: int,
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the Williams %R for given period over the input data.
    """
    try:
        highh = data[ohlcv_cols[1]].rolling(window=period).max()
        lowl = data[ohlcv_cols[2]].rolling(window=period).min()
        return -100 * ((highh - data[ohlcv_cols[3]]) / (highh - lowl))
    except Exception as e:
        raise CustomException(e, sys)


def cci(
    data: pd.DataFrame,
    period: int = 40,
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the commodity channel index (CCI) for given period over the input data.
    """
    try:
        TP = (data[ohlcv_cols[1]] + data[ohlcv_cols[2]] + data[ohlcv_cols[3]]) / 3
        SMA = sma(data, period=period, column=ohlcv_cols[3])
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
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the true range (TR) over the input data.
    """
    try:
        high, low, prev_close = (
            data[ohlcv_cols[1]],
            data[ohlcv_cols[2]],
            data[ohlcv_cols[3]].shift(),
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
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the average true range (ATR) over the input data for a particular period of time.
    """
    try:
        return rma(tr(data, ohlcv_cols), period)
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
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the directional index (DI) over the input data for a particular period of time.
    """
    try:
        di_plus = 100 * (
            data[ohlcv_cols[1]].diff().ewm(span=period, adjust=False).mean()
            / atr(data, period=period, ohlcv_cols=ohlcv_cols)
        )
        di_minus = 100 * (
            data[ohlcv_cols[2]]
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
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the average directional index (ADX) over the input data for a particular period of time.
    """
    try:
        DI = di(data, period=period, ohlcv_cols=ohlcv_cols)
        return ((DI.shift(1) * (period - 1)) + DI) / period
    except Exception as e:
        raise CustomException(e, sys)


def psar(
    data: pd.DataFrame,
    af: float = 0.02,
    max: float = 0.2,
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the PSAR (Percentage Strengthening) for a given stock.
    """
    try:
        AF = np.full(len(data), np.nan)
        AF[0] = af
        PSAR = np.full(len(data), np.nan)
        PSAR[0] = data.iloc[0][ohlcv_cols[2]]
        EP = np.full(len(data), np.nan)
        EP[0] = data.iloc[0][ohlcv_cols[1]]
        PSARdir = ["bull"] + [""] * (len(data) - 1)

        for i in range(1, len(data)):  # start on second data row
            prev_i = i - 1
            if PSARdir[prev_i] == "bull":
                PSAR[i] = PSAR[prev_i] + (AF[prev_i] * (EP[prev_i] - PSAR[prev_i]))
                PSARdir[i] = "bull"

                if (data.iloc[i][ohlcv_cols[2]] < PSAR[prev_i]) or (
                    data.iloc[i][ohlcv_cols[2]] < PSAR[i]
                ):
                    PSARdir[i] = "bear"
                    PSAR[i] = EP[prev_i]
                    EP[i] = data.iloc[prev_i][ohlcv_cols[2]]
                    AF[i] = af
                else:
                    if data.iloc[i][ohlcv_cols[1]] > EP[prev_i]:
                        EP[i] = data.iloc[i][ohlcv_cols[1]]
                        AF[i] = min(max, AF[prev_i] + af)
                    else:
                        AF[i] = AF[prev_i]
                        EP[i] = EP[prev_i]

            else:
                PSAR[i] = PSAR[prev_i] - (AF[prev_i] * (PSAR[prev_i] - EP[prev_i]))
                PSARdir[i] = "bear"

                if (
                    data.iloc[i][ohlcv_cols[1]] > PSAR[prev_i]
                    or data.iloc[i][ohlcv_cols[1]] > PSAR[i]
                ):
                    PSARdir[i] = "bull"
                    PSAR[i] = EP[prev_i]
                    EP[i] = data.iloc[prev_i][ohlcv_cols[1]]
                    AF[i] = af
                else:
                    if data.iloc[i][ohlcv_cols[2]] < EP[prev_i]:
                        EP[i] = data.iloc[i][ohlcv_cols[2]]
                        AF[i] = min(max, AF[prev_i] + af)
                    else:
                        AF[i] = AF[prev_i]
                        EP[i] = EP[prev_i]

        return pd.Series(PSAR, index=data.index)
    except Exception as e:
        raise CustomException(e, sys)


# def overnight_percent_diff(
#     data: pd.DataFrame, open_close_cols: Tuple[str, str] = ("Open", "Close")
# ) -> pd.Series:
#     """
#     Calculates the percent change overnight during market closed hours.
#     """
#     try:
#         return 100 * (
#             (data[open_close_cols[0]] - data[open_close_cols[1]].shift(1))
#             / data[open_close_cols[1]].shift(1)
#         )
#     except Exception as e:
#         raise CustomException(e, sys)


def price_volume(
    data: pd.DataFrame,
    ohlcv_cols: Tuple[str, str, str] = ("Open", "High", "Low", "Close", "Volume"),
) -> pd.Series:
    """
    Calculates the price-volume ratio (PVR) over the input data.
    """
    try:
        avg_price = data[ohlcv_cols[:-1]].mean(axis=1)
        return data[ohlcv_cols[2]] * avg_price / 1000.0
    except Exception as e:
        raise CustomException(e, sys)


def get_technical_indicators(
    df: pd.DataFrame,
    ohlcv_cols: Tuple[str, str, str, str] = (
        "High",
        "Low",
        "Close",
        "Open",
        "Volume",
    ),
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    try:
        df_copy = df.copy()
        df_copy["PV"] = price_volume(df_copy, ohlcv_cols=ohlcv_cols)
        df_copy["MACD"] = macd(df_copy, ohlcv_cols[3])
        df_copy["TR"] = tr(df_copy, ohlcv_cols=ohlcv_cols)
        df_copy["ATR"] = atr(df_copy, period=14, ohlcv_cols=ohlcv_cols)
        df_copy["RSI"] = rsi(df_copy, period=14, column=ohlcv_cols[3])
        df_copy["Momentum"] = momentum(df_copy, shift=14, column=ohlcv_cols[3])
        df_copy["PSAR"] = psar(df_copy, af=0.02, max=0.2, ohlcv_cols=ohlcv_cols)
        df_copy["CCI"] = cci(df_copy, period=20, ohlcv_cols=ohlcv_cols)
        df_copy["SMA"] = sma(df_copy, period=20, column=ohlcv_cols[3])
        df_copy["WMA"] = wma(df_copy, period=20, column=ohlcv_cols[3])
        df_copy["HMA"] = hma(df_copy, period=20, column=ohlcv_cols[3])
        df_copy["ADX"] = adx(df_copy, period=14, ohlcv_cols=ohlcv_cols)
        df_copy["Williams_R"] = williams_r(df_copy, period=14, ohlcv_cols=ohlcv_cols)
        log_params = [
            ("Close", "Close", 0, 1),
            ("Close", "Close", 1, 2),
            ("Close", "Close", 2, 3),
            ("Close", "Close", 3, 4),
            ("High", "Open", 0, 0),
            ("High", "Open", 0, 1),
            ("High", "Open", 0, 2),
            ("High", "Open", 0, 3),
            ("High", "Open", 1, 1),
            ("High", "Open", 2, 2),
            ("High", "Open", 3, 3),
            ("Low", "Open", 0, 0),
            ("Low", "Open", 1, 1),
            ("Low", "Open", 2, 2),
            ("Low", "Open", 3, 3),
        ]
        i = 1
        for col1, col2, shift1, shift2 in log_params:
            df_copy[f"r{i}"] = log_price(df_copy, col1, col2, shift1, shift2)
            i += 1
        df_copy = stochastic_oscillator(df_copy, period=14, ohlcv_cols=ohlcv_cols)
        if inplace:
            df = df_copy
        return df_copy
    except Exception as e:
        raise CustomException(e, sys)
