# Model comparison

Dataset: IBM Telco. Seed: 42. Train / validation / test: 4225 / 1409 / 1409.

All rows below use threshold 0.5; the model was selected using validation ROC-AUC.

| Model | Validation AUC | Test AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.860 | 0.832 | 0.619 | 0.527 | 0.569 |
| Random Forest | 0.857 | 0.831 | 0.641 | 0.468 | 0.541 |
| XGBoost | 0.857 | 0.832 | 0.622 | 0.471 | 0.536 |

Selected: **Logistic Regression**, validation-tuned threshold **0.495**.
Final test precision 0.614, recall 0.532, F1 0.570, ROC-AUC 0.832.
Confusion matrix [[TN, FP], [FN, TP]]: `[[910, 125], [175, 199]]`.

Risk is High at or above the selected threshold; Low otherwise. Scores are model estimates, not calibrated guarantees.
No test-based retuning or refitting after threshold selection. Feature importance uses validation data and is associative, not causal.
