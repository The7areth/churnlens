# Business interpretation

The training partition contains 4,225 customers; 26.5% have the churn label. The model is therefore evaluated with precision/recall and ranking metrics rather than accuracy alone.

At the validation-selected threshold of 0.495, the selected model identifies 53.2% of actual churners in the held-out test set. Of customers flagged, 61.4% carry the churn label. The confusion matrix is `[[910, 125], [175, 199]]` in [[TN, FP], [FN, TP]] order.

A retention team could use this ranking to prioritize review, subject to outreach capacity and the cost of missed churners versus unnecessary contacts. No revenue or retention lift has been measured. A lower threshold typically reaches more churners and creates more false positives; this project's F1 objective is a demonstrable starting point, not a business optimum.

Training churn rates by contract: {"Month-to-month": 0.4305555555555556, "One year": 0.11049107142857142, "Two year": 0.02926829268292683}. These are associations in a fictional sample and do not show that switching contract causes lower churn.

Top validation permutation-importance rows (feature, mean ROC-AUC decrease, standard deviation):

```csv
tenure,0.1819241003384227,0.010690725476721114
Contract,0.03692345449378705,0.004546966667899027
InternetService,0.030094551654654043,0.00514492134710666
```

Correlated predictors can share importance. Validate on future data and test retention interventions before operational decisions.
