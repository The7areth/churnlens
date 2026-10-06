# Application walkthrough

## Score a customer

Start the API and Streamlit interface using [the setup guide](running-locally.md). Enter a customer’s account and service details, then select **Estimate churn risk**. The response shows the model’s churn score, risk label, decision threshold, and model provenance.

Changing an input and submitting again illustrates how the fitted model responds. These comparisons describe model behavior rather than the causal effect of changing a customer’s contract or services.

## Inspect model evidence

Open **Model evidence** to review the candidate models, evaluation metrics, and held-out results. Training-only preprocessing is persisted with the classifier. Model selection uses validation ROC-AUC; the operating threshold uses validation F1. The test partition provides the final reported evaluation.

The synthetic sample supports functional smoke tests. The IBM benchmark results are recorded separately in [the model comparison](../reports/comparison.md).

## Explore the API

Open FastAPI’s `/docs` page to inspect the typed request schema and submit the [example customer](../examples/customer.json). The UI calls the same API, which loads the saved preprocessing, classifier, and threshold. Invalid inputs receive a validation response; model readiness is exposed through `/health`.

## Continue into the implementation

Read the [complete walkthrough](project-walkthrough.md) for the data preparation, selection protocol, persistence, and serving flow. The [model card](model-card.md) describes the dataset, evaluation scope, and deployment requirements.
