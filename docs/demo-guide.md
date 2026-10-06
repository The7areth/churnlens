# SYNC demo guide — Hareth Al-Fawaz

## Two-minute walkthrough

1. **Problem, 15 seconds:** “I built a small system to help a retention team prioritize customer review. It estimates churn likelihood from account and service information.”
2. **Working product, 40 seconds:** Open Streamlit, enter a customer, and click Estimate churn risk. Explain the estimated probability and threshold. Change the contract or tenure and resubmit to illustrate model behavior. Say explicitly that this is not proof an intervention would prevent churn.
3. **Evidence, 35 seconds:** Open Model evidence. Explain the three candidates, training-only preprocessing, validation selection, and final held-out evaluation. Be ready to describe one false positive and one false negative.
4. **Engineering, 20 seconds:** Open FastAPI `/docs`, show the typed input contract, and explain that the UI calls this endpoint. The persisted artifact includes preprocessing and the threshold, preventing inconsistent transformations.
5. **Next step, 10 seconds:** “For a real business I would validate on future periods, calibrate probabilities, and choose the threshold using retention costs and team capacity.”

## Before October 9

- Read the preprocessing and training code; reproduce the metrics yourself.
- Practice a complete demo without internet. Keep the CSV, model artifact, Python environment, and screenshots on your laptop.
- Add [The7areth/churnlens](https://github.com/The7areth/churnlens) to your CV and pin it on your profile.
- Use real benchmark scores from `reports/comparison.md`, never the synthetic smoke-test scores.
- Prepare a screenshot fallback and start both services before the event.

## Questions to prepare

**Why not just accuracy?** Most customers in this sample do not churn. A majority-class classifier can look accurate while finding no churners. Precision, recall, F1, and ROC-AUC show different aspects of the problem.

**Why a validation split?** Choosing a model or a threshold from the test set would bias the final performance estimate. Validation handles those decisions; test data is for reporting.

**Why persist preprocessing?** The API must apply exactly the same learned medians, scaling, and category mapping used during training.

**Why not always XGBoost?** Complexity does not guarantee better generalization. The winner is whichever model earns the highest validation ROC-AUC in this fixed comparison.

**Why F1?** It is a simple demo compromise between precision and recall. A real retention program should specify costs and capacity before choosing a threshold.

**What does a probability mean?** It is a model estimate from a fictional sample. We have not demonstrated probability calibration or real-business transferability.

**What did you build?** Describe the parts you have reviewed and can explain: data validation, leakage-safe preprocessing, model comparison, persistence, API, UI, tests, and Docker configuration. Be transparent about AI assistance if asked.

## CV bullet

Built an end-to-end customer churn prediction system using Python, scikit-learn, XGBoost, FastAPI, Streamlit, and Docker; compared three models on 7,043 IBM Telco records, achieving 0.832 held-out ROC-AUC and 0.570 F1 with Logistic Regression, with reproducible preprocessing and a validated prediction API.

Use this after reviewing and reproducing the implementation. Docker configuration is included; local Docker runtime verification is recorded separately in `reports/verification.md`.
