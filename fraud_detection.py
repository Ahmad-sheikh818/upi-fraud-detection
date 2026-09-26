"""
fraud_detection.py
------------------
Runs three anomaly/fraud detection methods on upi_transactions.csv and
evaluates each against the synthetic ground-truth labels (is_fraud).

Methods:
  A. IQR-based outlier detection on transaction amounts
     (classic statistical rule: amount > Q3 + 1.5 * IQR)
  B. Isolation Forest (sklearn) on engineered features:
     log_amount, hour_of_day, sender transaction count, sender total amount,
     one-hot merchant category / device / city
  C. Time-series anomaly detection: hourly transaction counts aggregated
     over the 14-day window, rolling z-score flags anomalous hours; every
     transaction inside a flagged hour is marked suspicious.

Outputs:
  - Prints precision / recall / F1 for each method vs the is_fraud labels
  - Saves 4 charts to ../images/

Usage:
    python3 fraud_detection.py        (run from the python/ folder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "upi_transactions.csv")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def prf(y_true, y_pred):
    """Precision, recall, F1 for the fraud (positive) class."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "flagged": int(y_pred.sum())}


def show(name, m):
    print(f"{name}")
    print(f"  flagged={m['flagged']:>5}  "
          f"precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  "
          f"F1={m['f1']:.3f}")


# ----------------------------------------------------------------------------
# Method A — IQR on amounts
# ----------------------------------------------------------------------------
def method_iqr_amount(df):
    q1 = df["amount"].quantile(0.25)
    q3 = df["amount"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    print(f"[IQR] Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}  "
          f"upper fence={upper:,.0f}")
    return (df["amount"] > upper).astype(int)


# ----------------------------------------------------------------------------
# Method B — Isolation Forest on engineered features
# ----------------------------------------------------------------------------
def method_isolation_forest(df):
    sender_stats = df.groupby("sender_id")["amount"].agg(
        sender_txn_count="size", sender_total_amount="sum")
    feat = df[["sender_id"]].merge(sender_stats, on="sender_id",
                                   how="left").copy()
    feat["log_amount"] = np.log1p(df["amount"])
    feat["hour_of_day"] = df["hour_of_day"]
    # cyclical encoding of hour so 23 and 0 are neighbours
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour_of_day"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour_of_day"] / 24)

    cat = pd.DataFrame(
        OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        .fit_transform(df[["merchant_category", "device_type", "city"]]),
        index=df.index)
    X = pd.concat([feat[["log_amount", "hour_sin", "hour_cos",
                         "sender_txn_count", "sender_total_amount"]],
                   cat], axis=1)
    X.columns = X.columns.astype(str)  # sklearn needs uniform column types

    contamination = float(df["is_fraud"].mean())  # ~0.025
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=RANDOM_SEED, n_jobs=-1)
    pred = iso.fit_predict(X)          # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X)  # lower = more anomalous
    return (pred == -1).astype(int), scores


# ----------------------------------------------------------------------------
# Method C — time-series rolling z-score on hourly counts
# ----------------------------------------------------------------------------
def method_timeseries(df):
    ts = df.set_index("timestamp")
    hourly = ts.resample("h").size().rename("txn_count").to_frame()
    hourly["amount_sum"] = ts["amount"].resample("h").sum()

    # Fraud moves unusual money, so detect on hourly amount totals.
    # Compare each hour against the same hour-of-day across the 14 days
    # (a 2 AM spike is anomalous even if it is smaller than daytime peaks).
    hourly["hod"] = hourly.index.hour
    hod_mean = hourly.groupby("hod")["amount_sum"].transform("mean")
    hod_std = hourly.groupby("hod")["amount_sum"].transform("std")
    hod_std = hod_std.replace(0, np.nan).fillna(hourly["amount_sum"].std())
    hourly["z"] = (hourly["amount_sum"] - hod_mean) / hod_std
    hourly["hour_flagged"] = (hourly["z"].abs() > 3).astype(int)

    flagged_hours = hourly.index[hourly["hour_flagged"] == 1]
    hour_key = df["timestamp"].dt.floor("h")
    y_pred = hour_key.isin(flagged_hours).astype(int)
    return y_pred, hourly


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def chart_amount_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    legit = df.loc[df["is_fraud"] == 0, "amount"]
    fraud = df.loc[df["is_fraud"] == 1, "amount"]
    bins = np.logspace(np.log10(10), np.log10(100_000), 60)
    ax.hist(legit, bins=bins, alpha=0.7, label=f"Legit (n={len(legit):,})",
            color="#2b7bbb")
    ax.hist(fraud, bins=bins, alpha=0.8, label=f"Fraud (n={len(fraud):,})",
            color="#d62728")
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (INR, log scale)")
    ax.set_ylabel("Transactions")
    ax.set_title("Transaction amount distribution: fraud vs legitimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "amount_distribution.png"))
    plt.close(fig)


def chart_iforest_scores(df, scores):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores[df["is_fraud"] == 0], bins=60, alpha=0.7,
            label="Legit", color="#2b7bbb")
    ax.hist(scores[df["is_fraud"] == 1], bins=60, alpha=0.8,
            label="Fraud", color="#d62728")
    ax.set_xlabel("Isolation Forest anomaly score (lower = more anomalous)")
    ax.set_ylabel("Transactions")
    ax.set_title("Isolation Forest anomaly score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "anomaly_scores.png"))
    plt.close(fig)


def chart_fraud_by_hour(df):
    hourly = df.groupby("hour_of_day")["is_fraud"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(hourly.index, hourly["mean"] * 100, color="#d62728",
                  alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by hour of day")
    for b, n in zip(bars, hourly["size"]):
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{b.get_height():.1f}%", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fraud_rate_by_hour.png"))
    plt.close(fig)


def chart_timeseries(hourly):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hourly.index, hourly["txn_count"], color="#2b7bbb", lw=1,
            label="Hourly transaction count")
    flagged = hourly[hourly["hour_flagged"] == 1]
    ax.scatter(flagged.index, flagged["txn_count"], color="#d62728", s=36,
               zorder=5, label=f"Flagged anomaly (|z|>3, n={len(flagged)})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions per hour")
    ax.set_title("Hourly transaction volume with time-series anomalies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "time_series_anomalies.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    y_true = df["is_fraud"]
    print(f"Loaded {len(df):,} transactions, "
          f"{int(y_true.sum())} labelled fraud "
          f"({y_true.mean():.2%})\n")

    # ---- Method A: IQR ----
    pred_a = method_iqr_amount(df)
    m_a = prf(y_true, pred_a)
    print()
    show("Method A — IQR on amounts", m_a)

    # ---- Method B: Isolation Forest ----
    pred_b, scores = method_isolation_forest(df)
    m_b = prf(y_true, pred_b)
    print()
    show("Method B — Isolation Forest (engineered features)", m_b)

    # ---- Method C: time-series rolling z-score ----
    pred_c, hourly = method_timeseries(df)
    m_c = prf(y_true, pred_c)
    print(f"[TimeSeries] flagged hours: {int(hourly['hour_flagged'].sum())} "
          f"of {len(hourly)}")
    print()
    show("Method C — Time-series z-score (|z|>3 vs hour-of-day baseline)",
         m_c)

    # ---- Charts ----
    print("\nSaving charts...")
    chart_amount_distribution(df)
    chart_iforest_scores(df, scores)
    chart_fraud_by_hour(df)
    chart_timeseries(hourly)
    print("Charts saved to ../images/: amount_distribution.png, "
          "anomaly_scores.png, fraud_rate_by_hour.png, "
          "time_series_anomalies.png")

    # ---- Summary table ----
    print("\n=== SUMMARY (vs synthetic ground truth) ===")
    summary = pd.DataFrame(
        {"Method": ["A: IQR (amounts)",
                    "B: Isolation Forest",
                    "C: Time-series z-score"],
         "Flagged": [m_a["flagged"], m_b["flagged"], m_c["flagged"]],
         "Precision": [m_a["precision"], m_b["precision"], m_c["precision"]],
         "Recall": [m_a["recall"], m_b["recall"], m_c["recall"]],
         "F1": [m_a["f1"], m_b["f1"], m_c["f1"]]})
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
