# Insider trading signal classifier
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

Target will be adjusted againts market performance (S&P500) between t and t+63.

## Project setup 
## Create Python environment
- Install python 3 if needed.
- Create environment with 
```
python3 -m venv insider-trading-signal-env
```

### Run in VS code
- Open project in VS code
- In the UI, pick the kernel: insider-trading-signal

### Run in Jupyter
- Use Python environment with
```
source insider-trading-signal-env/bin/activate
```
- Update pip
```
pip install --upgrade pip
```
- Open Jupyter notebooks
- Exit Python environment with
```
deactivate
```

## Using parquets
```
pip install pyarrow
```

## Data fetching
```
pip install yfinance requests-cache pandas_market_calendars
```

## Using tensorflow
As of November 2025 Tensorflow does not support Python 3.13. We need to use Python 3.11. 
- Install Python 3.11
```
brew install python@3.11
```
- Confirm installation
```
python3.11 --version
```
- Create environment for Tensorflow
```
python3.11 -m venv tf-env
source tf-env/bin/activate
pip install --upgrade pip
pip install tensorflow   # on Intel Mac
# or: pip install tensorflow-macos   # on Apple silicon