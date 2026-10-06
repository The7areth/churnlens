# Roadmap

The current release demonstrates a complete local ML application. These are future improvements, not implemented features.

1. **Evaluation stability:** define a new evaluation plan before further tuning; use stratified cross-validation on development data, measure uncertainty, and preserve a genuinely unseen final dataset.
2. **Business operating point:** measure outreach capacity, intervention cost, churn loss, and intervention effectiveness; compare threshold policies and precision/recall at top-k customers.
3. **Calibration:** assess a reliability curve and fit calibration only on development data. Evaluate it on untouched data before treating scores as calibrated probabilities.
4. **Data quality:** investigate domain-aware handling of zero-tenure billing records and missingness indicators; quantify any benefit through development-only experiments.
5. **Responsible validation:** evaluate subgroup performance and whether demographic inputs should be excluded. Validate on representative future-period customer data with a defined horizon.
6. **Operational deployment:** add authentication, request/response monitoring without unnecessary personal data, rate limits, artifact integrity checks, health monitoring, and model/data drift alerts.
7. **Demonstration:** host a public sample-only demo and add a short walkthrough video after selecting a hosting platform and budget.

Retain a compact, understandable application. Add infrastructure only when a concrete requirement justifies it.
