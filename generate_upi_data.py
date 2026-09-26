"""
generate_upi_data.py
--------------------
Generates 10,000 synthetic UPI transactions with realistic fields and
labelled fraud cases (~2.5% fraud rate).

Fraud patterns injected (labelled in `fraud_pattern` column):
  - odd_hour_high_amount : transaction between 12am-5am with a high amount
  - round_number         : suspiciously round high-value amount (e.g. 50000)
  - velocity_burst       : one sender firing many transactions within minutes
  - new_device_high      : high amount from a device marked as new/untrusted

Legitimate transactions follow realistic behaviour:
  - amounts follow a log-normal distribution (most UPI spends are small)
  - activity peaks during daytime/evening hours
  - common merchant categories, Indian cities, Android/iOS devices

Usage:
    python3 generate_upi_data.py
Output:
    upi_transactions.csv  (in the same folder as this script)
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TXNS = 10_000
FRAUD_RATE = 0.025  # ~2.5%

np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------------
MERCHANT_CATEGORIES = [
    "Grocery", "Food & Dining", "Travel", "Mobile Recharge", "Shopping",
    "Utility Bills", "Entertainment", "P2P Transfer", "Healthcare", "Fuel",
]
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Patna", "Lucknow", "Indore",
]
CITY_WEIGHTS = np.array([0.16, 0.15, 0.13, 0.10, 0.09, 0.08, 0.08, 0.06,
                         0.05, 0.04, 0.03, 0.03])
CITY_WEIGHTS /= CITY_WEIGHTS.sum()

DEVICE_TYPES = ["Android", "iOS"]
DEVICE_WEIGHTS = [0.82, 0.18]

# Daytime-heavy hourly activity profile (probability of a txn in each hour)
HOUR_WEIGHTS = np.array([
    0.004, 0.002, 0.001, 0.001, 0.001, 0.002,   # 00-05 (quiet night)
    0.008, 0.020, 0.045, 0.065, 0.075, 0.080,   # 06-11 (morning ramp)
    0.085, 0.080, 0.070, 0.065, 0.060, 0.065,   # 12-17 (afternoon)
    0.075, 0.080, 0.070, 0.050, 0.030, 0.012,   # 18-23 (evening peak)
])
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()

SENDERS = [f"USER{100000 + i}" for i in range(3000)]
RECEIVERS = [f"MERCH{200000 + i}" for i in range(1500)] + \
            [f"USER{100000 + i}" for i in range(3000)]  # P2P receivers

START = pd.Timestamp("2026-09-01")


def random_timestamps(n, hour_weights=HOUR_WEIGHTS):
    """Random timestamps across 14 days, weighted by hour-of-day activity."""
    hours = np.random.choice(24, size=n, p=hour_weights)
    days = np.random.randint(0, 14, size=n)
    minutes = np.random.randint(0, 60, size=n)
    seconds = np.random.randint(0, 60, size=n)
    return (START + pd.to_timedelta(days, unit="D")
            + pd.to_timedelta(hours, unit="h")
            + pd.to_timedelta(minutes, unit="m")
            + pd.to_timedelta(seconds, unit="s"))


def make_legit(n):
    """Build n legitimate transactions."""
    ts = random_timestamps(n)
    # Log-normal amounts: median ~ Rs 350, long tail to a few thousand
    amounts = np.round(np.random.lognormal(mean=5.86, sigma=1.05, size=n), 2)
    amounts = np.clip(amounts, 10, 25_000)
    df = pd.DataFrame({
        "txn_id": [f"TXN{900000 + i}" for i in np.random.choice(
            range(10_000_000), size=n, replace=False)],
        "timestamp": ts,
        "sender_id": np.random.choice(SENDERS, size=n),
        "receiver_id": np.random.choice(RECEIVERS, size=n),
        "amount": amounts,
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, size=n),
        "device_type": np.random.choice(DEVICE_TYPES, size=n, p=DEVICE_WEIGHTS),
        "city": np.random.choice(CITIES, size=n, p=CITY_WEIGHTS),
        "is_fraud": 0,
        "fraud_pattern": "",
    })
    df["hour_of_day"] = df["timestamp"].dt.hour
    return df


def make_fraud(n):
    """Build n fraudulent transactions across four realistic patterns."""
    patterns = np.random.choice(
        ["odd_hour_high_amount", "round_number", "velocity_burst",
         "new_device_high"],
        size=n, p=[0.30, 0.25, 0.25, 0.20])
    rows = []
    burst_senders = np.random.choice(SENDERS, size=max(1, n // 8),
                                     replace=False)
    for i, pattern in enumerate(patterns):
        if pattern == "odd_hour_high_amount":
            hour = np.random.randint(0, 5)
            ts = START + pd.to_timedelta(np.random.randint(0, 14), unit="D") \
                + pd.to_timedelta(hour, unit="h") \
                + pd.to_timedelta(np.random.randint(0, 60), unit="m")
            amount = round(float(np.random.uniform(8_000, 60_000)), 2)
            device = np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
        elif pattern == "round_number":
            ts = random_timestamps(1)[0]
            amount = float(np.random.choice(
                [10_000, 20_000, 25_000, 50_000, 75_000, 99_999, 100_000]))
            device = np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
        elif pattern == "velocity_burst":
            # many quick small-to-mid transfers from one sender
            sender = np.random.choice(burst_senders)
            base = START + pd.to_timedelta(np.random.randint(0, 14), unit="D") \
                + pd.to_timedelta(np.random.randint(0, 24), unit="h")
            ts = base + pd.to_timedelta(np.random.randint(0, 30), unit="m")
            amount = round(float(np.random.uniform(500, 9_000)), 2)
            device = np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
            rows.append({
                "txn_id": f"TXN{910000 + i}",
                "timestamp": ts,
                "sender_id": sender,
                "receiver_id": np.random.choice(RECEIVERS),
                "amount": amount,
                "merchant_category": "P2P Transfer",
                "device_type": device,
                "city": np.random.choice(CITIES, p=CITY_WEIGHTS),
                "is_fraud": 1,
                "fraud_pattern": pattern,
                "hour_of_day": ts.hour,
            })
            continue
        else:  # new_device_high
            ts = random_timestamps(1)[0]
            amount = round(float(np.random.uniform(15_000, 80_000)), 2)
            device = "New Device"
        rows.append({
            "txn_id": f"TXN{910000 + i}",
            "timestamp": ts,
            "sender_id": np.random.choice(SENDERS),
            "receiver_id": np.random.choice(RECEIVERS),
            "amount": amount,
            "merchant_category": np.random.choice(MERCHANT_CATEGORIES),
            "device_type": device,
            "city": np.random.choice(CITIES, p=CITY_WEIGHTS),
            "is_fraud": 1,
            "fraud_pattern": pattern,
            "hour_of_day": ts.hour,
        })
    return pd.DataFrame(rows)


def main():
    n_fraud = int(N_TXNS * FRAUD_RATE)
    n_legit = N_TXNS - n_fraud

    legit = make_legit(n_legit)
    fraud = make_fraud(n_fraud)

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Re-sequence txn ids chronologically for a clean file
    df["txn_id"] = [f"TXN{100001 + i}" for i in range(len(df))]

    cols = ["txn_id", "timestamp", "sender_id", "receiver_id", "amount",
            "merchant_category", "device_type", "city", "hour_of_day",
            "is_fraud", "fraud_pattern"]
    df = df[cols]

    out = "upi_transactions.csv"
    df.to_csv(out, index=False)

    print(f"Wrote {out}: {len(df):,} rows")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%} "
          f"({int(df['is_fraud'].sum())} fraud txns)")
    print("\nFraud pattern breakdown:")
    print(df.loc[df["is_fraud"] == 1, "fraud_pattern"]
          .value_counts().to_string())


if __name__ == "__main__":
    main()
