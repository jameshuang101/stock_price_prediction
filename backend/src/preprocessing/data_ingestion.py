import yfinance as yf
from typing import List, Optional
import pandas as pd
import fredapi
from datetime import date, datetime, timedelta
import dateparser
import pickle
import json
import sys
from src.exception import CustomException
import pandas_market_calendars as mcal


# Relevant tickers: ^VIX, DX-Y.NYB, AAPL, MSFT, GOOG, AMZN, SPY, VOO, ^GSPC
def get_stock_data(
    ticker: str,
    start_date=None,
    end_date=None,
    period: Optional[str] = None,
    features: List[str] = ["Open", "High", "Low", "Close", "Volume"],
    as_json: bool = False,
    save_as: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retrieves select stock data (OHLCV) from the Yahoo Finance API between a given start and end date.

    Args:
        ticker (Any, required): NASDAQ ticker for the stock. Can list multiple in a single string, like "SPY AAPL".
        start_date (Any, optional): Start date as datetime object or string, like "2023-01-01".
        end_date (Any, optional): Start date as datetime object or string, like "2023-12-31".
        period (Any, optional): How long of a period in the past to get data from, as an alternative to specifying a date range, like "1mo".
        features (List[str], optional): Which stock features to return. Defaults to OHLCV.
        save_as (str, optional): File path to save results to.

    Returns:
        (pd.DataFrame): Select stock data in DataFrame format.

    Usage:
        aapl_data_2023 = get_stock_data(
            "AAPL ^VIX DX-Y.NYB",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        msft_data_1yr = get_stock_data(
            "MSFT ^VIX DX-Y.NYB",
            period="1yr",
            features=["Close"],
        )
    """
    try:
        # Fetch the data from yfinance
        if start_date and end_date:
            # Grab data between specified start and end date
            stock_data = yf.download(ticker, start=start_date, end=end_date)
        elif start_date and not end_date:
            # Grab data between specified start date and now
            stock_data = yf.download(ticker, start=start_date)
        elif period is not None:
            # Grab data up to a certain period in the past
            stock_data = yf.download(ticker, period=period)
        else:
            # Grab all historical data available
            stock_data = yf.download(ticker)

        # Extract the relevant columns
        res_df = stock_data[features]
        res_df.columns = features

    except Exception as e:
        raise CustomException(e, sys)

    if save_as:
        try:
            if save_as.endswith(".json"):
                with open(save_as, "w") as f:
                    f.write(res_df.to_json(orient="index", date_unit="s"))
            elif save_as.endswith((".pickle", ".pkl")):
                res_df.to_pickle(save_as)
            elif save_as.endswith(".csv"):
                res_df.to_csv(save_as)
            else:
                print(
                    "Invalid extension. Should be one of (.json, .pickle, .pkl, .csv)"
                )
                pass
        except Exception as e:
            raise CustomException(e, sys)

    if as_json:
        return res_df.to_json(orient="index", date_unit="s")
    return res_df


# Relevant tickers: EFFR, UNRATE, UMCSENT
def get_macro_data(
    fred_api_key: str,
    ticker: str = "EFFR UNRATE UMCSENT ^VIX DX-Y.NYB",
    start_date: Optional[str | datetime] = None,
    end_date: Optional[str | datetime] = None,
    as_json: bool = False,
    save_as: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retrieves macroeconomic data from the FRED and/or Yahoo Finance API between a given start and end date.

    Args:
        fred_api_key (str, optrional): Your Fred API key.
        ticker (str, optional): Ticker name to retrieve. Defaults to "EFFR UNRATE UMCSENT ^VIX DX-Y.NYB".
        start_date (Any, optional): Start date as datetime object or string, like "2023-01-01".
        end_date (Any, optional): Start date as datetime object or string, like "2023-12-31".
        save_as (str, optional): File path to save results to.

    Returns:
        (pd.DataFrame): Select macroeconomic data in DataFrame format.

    Usage:
        aapl_data_2023 = get_stock_data(
            "EFFR UNRATE UMCSENT",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
    """
    try:
        # Connect to Fred API
        fr = fredapi.Fred(api_key=fred_api_key)
        # Parse the ticker
        tickers = ticker.split(" ")
        # Dataframe to hold all results
        res_df = pd.DataFrame()
        # Grab each series individually and merge into dataframe

        if type(start_date) == str:
            start_date = dateparser.parse(start_date)

        if type(end_date) == str:
            end_date = dateparser.parse(end_date)

        for tk in tickers:
            # Try to grab from FRED API first
            if check_available_yf(tk) == False:
                if start_date and end_date:
                    res = fr.get_series(
                        series_id=tk,
                        observation_start=start_date.replace(day=1),
                        observation_end=end_date,
                    )
                elif start_date and not end_date:
                    res = fr.get_series(
                        series_id=tk,
                        observation_start=start_date.replace(day=1),
                        observation_end=date.today(),
                    )
                else:
                    res = fr.get_series(tk)

            # Try to grab from Yahoo Finance API
            else:
                res = get_stock_data(
                    tk, start_date=start_date, end_date=end_date, features=["Close"]
                )["Close"]

            # Name the column then merge into result dataframe
            res.rename(tk, inplace=True)
            res_df = res_df.merge(res, how="outer", left_index=True, right_index=True)

        # Forward fill and NaNs, then backwards fill to get first row NaNs
        res_df.ffill(axis=0, inplace=True)
        res_df.bfill(axis=0, inplace=True)

        if save_as:
            if save_as.endswith(".json"):
                with open(save_as, "w") as f:
                    f.write(res_df.to_json(orient="index", date_unit="s"))
            elif save_as.endswith((".pickle", ".pkl")):
                res_df.to_pickle(save_as)
            elif save_as.endswith(".csv"):
                res_df.to_csv(save_as)
            else:
                print(
                    "Invalid extension. Should be one of (.json, .pickle, .pkl, .csv)"
                )
                pass

        if as_json:
            return res_df.to_json(orient="index", date_unit="s")
        return res_df

    except Exception as e:
        raise CustomException(e, sys)


def check_available_yf(asset: str) -> bool:
    """
    Checks if an asset is available via the Yahoo Finance API.
    """
    try:
        info = yf.Ticker(asset).history(period="5d", interval="1d")
    except Exception as e:
        return False
    return len(info) > 0


def is_market_day(
    date: Optional[str | datetime] = None,
    start_date: Optional[str | datetime] = None,
    end_date: Optional[str | datetime] = None,
) -> bool:
    """
    Checks if a given date is a market day.
    """
    if date is not None:
        try:
            result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
        except Exception as e:
            raise CustomException(e, sys)
    elif start_date is not None and end_date is not None:
        try:
            result = mcal.get_calendar("NYSE").schedule(
                start_date=start_date, end_date=end_date
            )
        except Exception as e:
            raise CustomException(e, sys)
    else:
        print("No date provided")
        return False
    return result.empty == False


def get_market_days(
    start_date: Optional[str | datetime],
    end_date: Optional[str | datetime],
):
    """
    Get the market days between two dates.
    """
    if start_date is None or end_date is None:
        print("No date provided")
        return []
    try:
        market_calendar = mcal.get_calendar("NYSE")
        market_days = market_calendar.valid_days(
            start_date=start_date, end_date=end_date
        )
        return market_days.to_pydatetime()
    except Exception as e:
        raise CustomException(e, sys)


def get_lead_days(date: Optional[str | datetime] = None, lead_days: int = 18):
    """
    Get the N market days of lead data for feature engineering immediately preceding a date (Defaults to 18 lead days).
    """
    if date is not None:
        if type(date) == str:
            date = dateparser.parse(date)
        start_date = date - timedelta(days=round((lead_days + 2) * 1.5))
        days_list = get_market_days(start_date=start_date, end_date=date)
        return days_list[-lead_days - 1 : -1]
    else:
        print("No date provided")
        return []
