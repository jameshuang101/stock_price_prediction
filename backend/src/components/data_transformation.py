import pandas as pd


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
    return ema(data, period=26, column=column) - ema


def rma(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Calculates the average true range (ATR) over the input data for a particular period of time.
    """
    high, low, prev_close = df["high"], df["low"], df["close"].shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)
    return rma(tr, length)
