import yfinance as yf
from typing import List, Optional
import pandas as pd


# Tickers: ^VIX, DX-Y.NYB, AAPL, MSFT, GOOG, AMZN, SPY, VOO, ^GSPC
def get_stock_data(
    ticker,
    start_date=None,
    end_date=None,
    period=None,
    features: List[str] = ["Open", "High", "Low", "Close", "Volume"],
) -> pd.DataFrame:
    """
    Retrieves select stock data (OHLCV) between a given start and end date.

    Args:
        ticker (Any): NASDAQ ticker for the stock. Can list multiple in a single string, like "SPY AAPL".
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
        )
    """
    # Fetch the data from yfinance
    if start_date and end_date:
        # Grab data between specified start and end date
        stock_data = yf.download(ticker, start=start_date, end=end_date)
    elif period:
        # Grab data up to a certain period in the past
        stock_data = yf.download(ticker, period=period)
    else:
        # Grab all historical data available
        stock_data = yf.download(ticker)

    # Extract the relevant columns
    relevant_data = stock_data[features]

    return relevant_data
