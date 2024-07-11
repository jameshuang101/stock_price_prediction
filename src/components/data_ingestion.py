import yfinance as yf
from typing import List, Optional
import pandas as pd
import fredapi


# Relevant tickers: ^VIX, DX-Y.NYB, AAPL, MSFT, GOOG, AMZN, SPY, VOO, ^GSPC
def get_stock_data(
    ticker,
    start_date=None,
    end_date=None,
    period=None,
    features: List[str] = ["Open", "High", "Low", "Close", "Volume"],
) -> pd.DataFrame:
    """
    Retrieves select stock data (OHLCV) from the Yahoo Finance API between a given start and end date.

    Args:
        ticker (Any, required): NASDAQ ticker for the stock. Can list multiple in a single string, like "SPY AAPL".
        start_date (Any, optional): Start date as datetime object or string, like "2023-01-01".
        end_date (Any, optional): Start date as datetime object or string, like "2023-12-31".
        period (Any, optional): How long of a period in the past to get data from, as an alternative to specifying a date range, like "1mo".
        features (List[str], optional): Which stock features to return. Defaults to OHLCV.

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
    # Fetch the data from yfinance
    if start_date and end_date:
        # Grab data between specified start and end date
        stock_data = yf.download(ticker, start=start_date, end=end_date)
    elif start_date and not end_date:
        # Grab data between specified start date and now
        stock_data = yf.download(ticker, start=start_date)
    elif period:
        # Grab data up to a certain period in the past
        stock_data = yf.download(ticker, period=period)
    else:
        # Grab all historical data available
        stock_data = yf.download(ticker)

    # Extract the relevant columns
    relevant_data = stock_data[features]

    return relevant_data


# Relevant tickers: EFFR, UNRATE, UMCSENT
def get_macro_data(ticker: str, start_date=None, end_date=None):
    """
    Retrieves macroeconomic data from the FRED API between a given start and end date

    Args:
        ticker (Any, required): Ticker name to retrieve.
        start_date (_type_, optional): _description_. Defaults to None.
        end_date (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Put your Fred api key here
    fr = fredapi.Fred(api_key="69a47122deb5402556701ea01fa82a91")
    # Parse the ticker
    tickers = ticker.split(" ")
    # Dataframe to hold all results
    res_df = pd.DataFrame()
    # Grab each series individually and merge into dataframe
    for tk in tickers:
        if start_date and end_date:
            res = fr.get_series(
                series_id=tk, observation_start=start_date, observation_end=end_date
            )
        else:
            res = fr.get_series(tk)
        # Name the series then merge into result dataframe
        res.rename(tk, inplace=True)
        res_df = res_df.merge(res, how="outer", left_index=True, right_index=True)

    # Forward fill and NaNs, then backwards fill to get first row NaNs
    res_df.ffill(axis=0, inplace=True)
    res_df.bfill(axis=0, inplace=True)

    return res_df
