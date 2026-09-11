"""
generate_data.py
-----------------
Generates a synthetic financial transaction dataset with realistic,
built-in fraud patterns for demonstrating an end-to-end fraud detection
pipeline. Real transaction data is confidential/proprietary, so this
script creates a stand-in dataset that mimics known real-world fraud
characteristics (rare class, odd-hour clustering, high velocity, new
devices, amount anomalies) so the modeling pipeline behaves realistically.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_TRANSACTIONS = 50_000
FRAUD_RATE = 0.017  # ~1.7% fraud, realistic for this kind of demo dataset

rng = np.random.default_rng(RNG_SEED)

def generate_dataset(n=N_TRANSACTIONS, fraud_rate=FRAUD_RATE, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    merchant_categories = ["Grocery", "Electronics", "Travel", "Dining",
                            "Fashion", "Utilities", "Entertainment", "Online Services"]

    def make_block(n_rows, is_fraud):
        if is_fraud:
            # Fraud: higher amounts, odd hours, high velocity, new devices more common
            # (heavy overlap with legitimate transactions is intentional & realistic —
            # real fraud is NOT perfectly separable from normal behaviour)
            amount = rng.gamma(shape=2.1, scale=1400, size=n_rows)
            hour = rng.choice(
                list(range(24)), size=n_rows,
                p=_hour_weights(fraud=True)
            )
            txn_velocity = rng.poisson(lam=2.6, size=n_rows)          # transactions in last hour
            account_age_days = rng.integers(1, 2500, size=n_rows)
            account_age_days = np.where(rng.random(n_rows) < 0.35,
                                         rng.integers(1, 200, size=n_rows), account_age_days)
            new_device_flag = rng.choice([0, 1], size=n_rows, p=[0.55, 0.45])
            distance_from_home_km = rng.gamma(shape=1.7, scale=70, size=n_rows)
        else:
            amount = rng.gamma(shape=2.0, scale=850, size=n_rows)
            hour = rng.choice(
                list(range(24)), size=n_rows,
                p=_hour_weights(fraud=False)
            )
            txn_velocity = rng.poisson(lam=1.1, size=n_rows)
            account_age_days = rng.integers(30, 3000, size=n_rows)
            new_device_flag = rng.choice([0, 1], size=n_rows, p=[0.90, 0.10])
            distance_from_home_km = rng.gamma(shape=1.4, scale=18, size=n_rows)

        merchant_category = rng.choice(merchant_categories, size=n_rows)
        # risk_score kept only as a rough illustrative summary (not fed into the model
        # itself, to avoid redundant/leaky features) — a noisy, partial combination
        risk_score = (
            0.3 * _minmax(amount) +
            0.3 * _minmax(txn_velocity) +
            0.2 * new_device_flag +
            0.2 * _minmax(distance_from_home_km)
        )
        risk_score = np.clip(risk_score + rng.normal(0, 0.12, n_rows), 0, 1)

        return pd.DataFrame({
            "amount": np.round(amount, 2),
            "hour": hour,
            "txn_velocity_last_hour": txn_velocity,
            "account_age_days": account_age_days,
            "new_device_flag": new_device_flag,
            "distance_from_home_km": np.round(distance_from_home_km, 1),
            "merchant_category": merchant_category,
            "risk_score": np.round(risk_score, 4),
            "is_fraud": int(is_fraud),
        })

    fraud_df = make_block(n_fraud, is_fraud=True)
    legit_df = make_block(n_legit, is_fraud=False)

    df = pd.concat([fraud_df, legit_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{100000+i}" for i in range(len(df))])

    # Inject a small amount of realistic missingness (device info sometimes missing)
    missing_idx = rng.choice(df.index, size=int(0.015 * len(df)), replace=False)
    df.loc[missing_idx, "distance_from_home_km"] = np.nan

    return df


def _hour_weights(fraud: bool):
    """Return a probability distribution over 24 hours."""
    hours = np.arange(24)
    if fraud:
        # Fraud skews toward late night / early morning
        weights = np.exp(-0.5 * ((hours - 2) / 4.5) ** 2) + 0.15
    else:
        # Legit skews toward daytime/evening
        weights = np.exp(-0.5 * ((hours - 15) / 6) ** 2) + 0.1
    return weights / weights.sum()


def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo == 0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    print(df["is_fraud"].value_counts(normalize=True))
