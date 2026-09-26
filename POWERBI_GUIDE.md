# Power BI Fraud Dashboard — Build Guide

Build an interactive UPI fraud monitoring dashboard from
`data/upi_transactions.csv`. Estimated build time: 30–45 minutes.

## 1. Connect the data

1. Power BI Desktop → **Get Data → Text/CSV** → select `upi_transactions.csv`.
2. In Power Query, set types:
   - `timestamp` → Date/Time
   - `amount` → Decimal Number
   - `hour_of_day`, `is_fraud` → Whole Number
   - everything else → Text
3. Add a calculated column for a readable flag:
   ```dax
   Txn Status = IF('upi_transactions'[is_fraud] = 1, "Fraud", "Legitimate")
   ```
4. Add a calendar column for time intelligence:
   ```dax
   Txn Date = DATE(YEAR('upi_transactions'[timestamp]),
                   MONTH('upi_transactions'[timestamp]),
                   DAY('upi_transactions'[timestamp]))
   ```

## 2. DAX measures

```dax
Total Transactions = COUNTROWS('upi_transactions')

Fraud Count = CALCULATE(
    COUNTROWS('upi_transactions'),
    'upi_transactions'[is_fraud] = 1
)

Fraud Rate % = DIVIDE([Fraud Count], [Total Transactions]) * 100

Total Fraud Amount = CALCULATE(
    SUM('upi_transactions'[amount]),
    'upi_transactions'[is_fraud] = 1
)

Avg Fraud Amount = CALCULATE(
    AVERAGE('upi_transactions'[amount]),
    'upi_transactions'[is_fraud] = 1
)

-- transactions per sender (for the velocity table)
Txns per Sender = COUNTROWS('upi_transactions')
```

## 3. Suggested pages & visuals

**Page 1 — Overview (KPI row + trends)**
- Cards: Total Transactions · Fraud Count · Fraud Rate % · Total Fraud Amount
- Line chart: Fraud Count by Txn Date (trend over the 14 days)
- Donut: Fraud vs Legitimate share

**Page 2 — Patterns**
- Bar chart: Fraud Rate % by hour_of_day (the night-time spike is the story)
- Bar chart: Fraud Count by fraud_pattern (odd-hour vs burst vs round-number…)
- Bar chart: Fraud Count by merchant_category
- Map or bar: Fraud Count by city

**Page 3 — Deep dive (analyst view)**
- Table: top senders by Txns per Sender + Total Fraud Amount,
  filtered to `is_fraud = 1` — this surfaces velocity-burst accounts
- Slicers: Txn Date, city, device_type, merchant_category
- Scatter: amount vs hour_of_day, coloured by Txn Status

## 4. Formatting tips

- Conditional formatting: colour the Fraud Rate % bar red above 10%.
- Add a text box on Page 2 noting the insight: *"Fraud concentrates
  between 12am–5am — up to 72% of night-hour volume."*
- Keep the palette to 2–3 colours (e.g. blue for legitimate, red for fraud)
  so the story reads instantly.

## 5. Publish / share

- Save as `upi_fraud_dashboard.pbix` next to this guide.
- Export the overview page as PDF for the portfolio write-up.
