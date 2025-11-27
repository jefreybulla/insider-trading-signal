# src/insider/calendar.py
import pandas as pd
from exchange_calendars import get_calendar

def nyse_days(start, end) -> pd.DatetimeIndex:
    nyse = get_calendar('XNYS')
    # Use sessions to get valid trading days
    sessions = nyse.sessions_in_range(start, end)
    return sessions

def next_trading_day_or_same(trade_days: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    pos = trade_days.searchsorted(t, side='left')
    if pos >= len(trade_days):
        return trade_days[-1]
    return trade_days[pos]

def plus_n_days(trade_days: pd.DatetimeIndex, t0: pd.Timestamp, n: int) -> pd.Timestamp:
    pos = trade_days.searchsorted(t0, side='left')
    pos_n = pos + n
    if pos_n >= len(trade_days):
        return trade_days[-1]
    return trade_days[pos_n]
