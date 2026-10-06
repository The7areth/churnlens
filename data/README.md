# Data provenance

The full benchmark is the [IBM Telco Customer Churn CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv), a fictional telecom sample with 7,043 records, 19 input fields, customerID, and the Yes/No target Churn. IBM's source repository is archived and distributed under Apache-2.0; retain its license with redistributed source data. The downloaded file is ignored by Git; retrieve it using `python -m churn.data`.

Expected SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

`sample.csv` contains 180 independently generated, synthetic fixture rows. It is included for offline smoke tests only. Its deliberately simplified label rule is not evidence of real predictive quality. Training reports and API output explicitly identify non-IBM data. The delivered archive also contains the verified IBM CSV for offline use.

Inputs: demographic categories (gender, senior citizen, partner, dependents); account tenure in months; phone/internet subscriptions and add-ons; contract and payment method; paperless billing; monthly and total charges. Blank TotalCharges is treated as missing and imputed from the training median. Customer ID is used for split auditing, never as a predictor. Churn is the target and never an input. No churn-reason, churn-score, or other outcome-derived fields are used.
