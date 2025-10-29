# Data mapping

Given the SEC insiders data, ese this info to find the relevant data for this project. Source: `FORM_345_readme.htm` 

## Model features
- reporting_owner: OFFICER, DIRECTOR, TENPERCENTOWNER, OTHER
- side: buy or sell
- dollar_value = shares * price_per_share
- size_vs_cap = dollar_value / market_cap_t-1
- is_10b5_1 = 0 or 1 (0 = open-market/discretionary, 1 = 10b5_1 plan)

### Computing side
- Use NONDERIV_TRANS file
- In SECURITY_TITLE find "COMMON" stock transactions
- Get TRANS_CODE in ('P','S')
    - Then map: P → Buy, S → Sell

### Regarding time of transation
In order to train our model we need to determine the price of the stock 63 trading days (~ 1 quarter) after the insider transaction. In the dateset `transaction_date` is the date of the transaction. It's not a model feature but we need it so we can find the stock price at `t + 63`.

## Date we need from the SEC dataset

## SUBMISSION.tsv

AFF10B5ONE maps to is_10b5_1

| Field Name            | Description                                                             |      Data Type | Nullable | Key |
| --------------------- | ----------------------------------------------------------------------- | -------------: | :------: | :-: |
| AFF10B5ONE | The transaction was made pursuant to a contract, instruction or written plan for the purchase or sale of equity securities of the issuer that is intended to satisfy the affirmative defense conditions of Rule 10b5-1(c). | VARCHAR2 (25) | Yes | |

### REPORTINGOWNER.tsv

RPTOWNER_RELATIONSHIP maps to reporting_owner

| Field Name            | Description                                                             |      Data Type | Nullable | Key |
| --------------------- | ----------------------------------------------------------------------- | -------------: | :------: | :-: |
| RPTOWNER_RELATIONSHIP | Reporting owner relationship OFFICER, DIRECTOR, TENPERCENTOWNER, OTHER. | VARCHAR2 (100) |    No    |     |


### NONDERIV_TRANS.tsv
TRANS_DATE maps to transaction_date
TRANS_SHARES is used to determine dollar_value
TRANS_PRICEPERSHARE is used to determine dollar_value


| Field Name            | Description                                                             |      Data Type | Nullable | Key |
| --------------------- | ----------------------------------------------------------------------- | -------------: | :------: | :-: |
| TRANS_DATE | Transaction date in (DD-MON-YYYY) format. | DATE |    No    |     |
| TRANS_SHARES | Transaction shares reported when Securities Acquired (A) or Disposed of (D). | NUMBER(16,2) | Yes  |     |
| TRANS_PRICEPERSHARE | Price of non-Derivative Transaction Security. | NUMBER(16,2) | YES |     |
| TRANS_CODE | Transaction code (values and descriptions are listed in the Appendix section 6.2 Trans Code List). | VARCHAR2 (1) | Yes | |