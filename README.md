# insider-trading-signal
Given an insider buy or sell (Form 4 SEC), predict whether the stock’s price will be up or down ~1 quarter later (return over the next 63 trading days).

## Model features
- reporting_owner: OFFICER, DIRECTOR, TENPERCENTOWNER, OTHER
- side: buy or sell
- dollar_value = shares * price_per_share
- size_vs_cap = dollar_value / market_cap_t-1
- is_10b5_1 = 0 or 1 (0 = open-market/discretionary, 1 = 10b5_1 plan)

10b5_1 is a pre-arranged trading plan. These purchases are often often less informative than open-market (discretionary) trades.

## Model target
Class = 1 (Up) if forward return > 0; Class = 0 (Down) otherwise. 

Target will be adjusted agains market performance (S&P500)
