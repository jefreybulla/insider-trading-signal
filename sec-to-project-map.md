# Data mapping

Given the SEC insiders data, ese this info to find the relevant data for this project. Source: `FORM_345_readme.htm` 

## Model features
- reporting_owner: OFFICER, DIRECTOR, TENPERCENTOWNER, OTHER
- side: buy or sell
- dollar_value
- size_vs_cap = dollar_value / market_cap_t-1
- is_10b5_1 = 0 or 1 (0 = open-market/discretionary, 1 = 10b5_1 plan)

10b5_1 is a pre-arranged trading plan. These purchases are often often less informative than open discretionary trades.

`transaction_date` is the date of the transaction. It's not a feature but we need so we can find the stock price at `t + 63` days.

## Location of features in SEC dataset

### REPORTINGOWNER.tsv

RPTOWNER_RELATIONSHIP: maps to reporting_owner

| Field Name            | Description                                                             |      Data Type | Nullable | Key |
| --------------------- | ----------------------------------------------------------------------- | -------------: | :------: | :-: |
| RPTOWNER_RELATIONSHIP | Reporting owner relationship OFFICER, DIRECTOR, TENPERCENTOWNER, OTHER. | VARCHAR2 (100) |    No    |     |


### NONDERIV_TRANS.tsv
TRANS_DATE: maps to transaction_date


| Field Name            | Description                                                             |      Data Type | Nullable | Key |
| --------------------- | ----------------------------------------------------------------------- | -------------: | :------: | :-: |
| TRANS_DATE | Transaction date in (DD-MON-YYYY) format. | DATE |    No    |     |