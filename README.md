# Financial Loan Risk Analysis

A machine learning pipeline predicting loan approval decisions for FinTech
Innovations, built around the business's actual cost of mistakes rather than
standard accuracy alone.

## Overview

FinTech Innovations currently relies on manual loan underwriting. This project
builds a model to automate part of that decision, using a leakage-safe
preprocessing pipeline and comparing five algorithms (Logistic Regression, KNN,
Random Forest, Extra Trees, Gradient Boosting) via `GridSearchCV` and
`RandomizedSearchCV`.

The final model — Logistic Regression, deployed at a cost-optimized decision
threshold of 0.92 rather than the default 0.5 — cuts the dollar cost of approval
mistakes by roughly 90% on held-out test data, using the business's own error
costs: $50,000 per bad approval, $8,000 per missed good applicant.

## Key Results

- **ROC-AUC:** 0.9786
- **Decision threshold:** 0.92 (optimized against dollar cost, not accuracy)
- **Cost reduction vs. naive baseline:** ~90%, roughly $9,454 saved per application
- **Top drivers of approval:** debt-to-income ratio, monthly income, net worth

## Approach

- CRISP-DM structure: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation
- Custom business-cost metric built around the $50K/$8K asymmetric error costs
- `ColumnTransformer` + `Pipeline` for leakage-safe preprocessing, including a
  custom transformer for engineered financial ratio features
- 5-fold cross-validation across all models; Gradient Boosting tuned with a
  two-stage search (broad `RandomizedSearchCV` → refined `GridSearchCV`)
- A brief secondary regression model (predicting `RiskScore`) included for
  comparison against the primary classification approach

## Files

- `financial_loan_risk.ipynb` — full analysis notebook

## Tools

Python, pandas, scikit-learn, matplotlib, seaborn