import math
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def remove_inf_and_nan(
    df: pd.DataFrame, behavior: str = "impute", window: int = 14
) -> Optional[pd.DataFrame]:
    """
    Removes infinite and NaN values from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        behavior (str, optional): Behavior when encountering infinite or NaN values. Options are "impute" or "drop" or "mean". Defaults to "impute".
        window (int, optional): Window size for imputation. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with infinite and NaN values handled.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    if behavior == "impute":
        for col in df.columns:
            inds = np.where(df[col].isna())[0]
            for i in inds:
                df.iloc[i][col] = df.iloc[
                    max(0, i - window // 2) : min(len(df), i + window // 2)
                ][col].mean()
    elif behavior == "mean":
        for col in df.columns:
            df.fillna(df[col].mean(), inplace=True)
    elif behavior == "drop":
        df.dropna(inplace=True)
    return df
