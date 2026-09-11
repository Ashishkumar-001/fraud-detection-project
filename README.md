# Fraud Detection in Digital Financial Transactions

An end-to-end machine learning project that detects fraudulent transactions in a
simulated digital payments dataset, built as the practical implementation of a
4-week Data Science internship (NSDC / Yuva Intern — Virtual Data Science Explorer).

## About the dataset

Real bank transaction data is confidential and not publicly accessible for a
project like this. This repository therefore includes a script
(`data/generate_data.py`) that generates a **synthetic transaction dataset**
(50,000 rows, ~1.7% fraud rate) with realistic, well-documented fraud
characteristics based on patterns commonly reported in fraud detection research:

- Fraud is rare (imbalanced classes), matching real-world class imbalance.
- Fraud transactions skew toward late-night / early-morning hours.
- Fraud shows higher transaction velocity (more transactions in a short window).
- Fraud is more common on newer devices and newer accounts.
- Fraud transactions tend to occur further from the account's usual location.
- Deliberate noise and class overlap are included so the problem is **not
  trivially/perfectly separable** — mirroring real-world fraud detection, where
  even good models make mistakes.

This lets the notebook demonstrate a genuine, honestly-evaluated ML pipeline
without requiring access to sensitive financial data.

## Project structure

```
fraud-detection-project/
├── data/
│   ├── generate_data.py       # generates the synthetic dataset
│   └── transactions.csv       # generated dataset (50,000 rows)
├── notebooks/
│   └── fraud_detection_analysis.ipynb   # full pipeline: EDA → preprocessing → modeling → evaluation
├── outputs/                   # saved charts from the notebook run
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── evaluation_results.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## What the notebook does

1. **EDA** — distribution plots, correlation heatmap, class balance analysis.
2. **Preprocessing** — missing value imputation, scaling, one-hot encoding, via an
   `sklearn` `ColumnTransformer` + `Pipeline`.
3. **Modeling** — trains and compares Logistic Regression, Random Forest, and
   XGBoost using 5-fold stratified cross-validation.
4. **Evaluation** — precision, recall, F1-score, ROC-AUC, confusion matrix, and
   ROC curve comparison. The best model is selected by **F1-score** rather than
   accuracy or ROC-AUC alone, since fraud is a rare-event classification problem
   where those metrics can be misleading.
5. **Feature importance** — identifies which signals matter most for flagging risk.

## Results (this run)

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC |
|---|---|---|---|---|
| Random Forest | ~0.50 | ~0.68 | ~0.58 | ~0.97 |
| XGBoost | ~0.37 | ~0.74 | ~0.50 | ~0.97 |
| Logistic Regression | ~0.18 | ~0.90 | ~0.30 | ~0.97 |

(Exact numbers vary slightly by run/seed — see the notebook for the live results.)

## How to run it

```bash
pip install -r requirements.txt
python data/generate_data.py
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

## Honest limitations

- Synthetic data, not real transactions — real-world performance would likely be
  lower and noisier.
- No hyperparameter tuning was run in this notebook (kept simple for the
  internship's scope); a tuning strategy is discussed in the project's written
  planning reports.
- This project is part of an academic/internship exercise, not a
  production-ready fraud detection system.

## Author

Ashish — B.Tech, submitted as part of the NSDC Virtual Data Science Explorer
Internship (Weeks 1–4) and as supporting project work for academic internship
evaluation.
