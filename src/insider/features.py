# src/insider/features.py
import numpy as np
import pandas as pd
from .calendar import next_trading_day_or_same, plus_n_days

def compute_forward_return(df_events: pd.DataFrame, px: pd.DataFrame, trade_days: pd.DatetimeIndex, n=63):
    """
    df_events: columns ['ticker','date'] (date tz-naive)
    px: wide Adj Close (date x tickers)
    """
    def pick_prices(row):
        tic, t = row["ticker"], pd.Timestamp(row["date"])
        try:
            t0 = next_trading_day_or_same(trade_days, t)
            t63 = plus_n_days(trade_days, t0, n)
        except (ValueError, IndexError):
            return pd.Series({"P_t": np.nan, "P_t_plus_63": np.nan})
        
        if tic not in px.columns:
            return pd.Series({"P_t": np.nan, "P_t_plus_63": np.nan})
        
        s = px[tic].dropna()
        if len(s) == 0:
            return pd.Series({"P_t": np.nan, "P_t_plus_63": np.nan})
        
        # align by nearest trading days in px
        pos0 = s.index.searchsorted(t0, "left")
        if pos0 >= len(s):
            pos0 = len(s) - 1
        p0 = s.iloc[pos0]
        
        pos63 = s.index.searchsorted(t63, "left")
        if pos63 >= len(s):
            pos63 = len(s) - 1
        p63 = s.iloc[pos63]
        
        if pd.isna(p0) or pd.isna(p63):
            return pd.Series({"P_t": np.nan, "P_t_plus_63": np.nan})
        return pd.Series({"P_t": float(p0), "P_t_plus_63": float(p63)})
    out = df_events.apply(pick_prices, axis=1)
    return pd.concat([df_events, out], axis=1)

def log1p_safe(x: pd.Series) -> pd.Series:
    return np.log1p(x.clip(lower=0))

def size_vs_cap(dollar_value: pd.Series, cap_tminus1: pd.Series) -> pd.Series:
    return (dollar_value / cap_tminus1).replace([np.inf, -np.inf], np.nan)