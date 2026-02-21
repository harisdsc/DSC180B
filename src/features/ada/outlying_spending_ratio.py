import pandas as pd

def outlying_spending_ratio(
    txn_df,
    consumer_col="prism_consumer_id",
    amt_col="amount",
    direction_col="credit_or_debit",
    debit_value="DEBIT",
    iqr_k=1.5
):
    """
    Computes:
        outlying_spending_ratio =
        (# debit txns above Q3 + k*IQR) / (# all debit txns)
    """

    df = txn_df.copy()

    # Keep only debit transactions
    df = df[df[direction_col] == debit_value].copy()
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
    df = df[df[amt_col].notna() & (df[amt_col] > 0)]

    if df.empty:
        return pd.DataFrame(columns=[
            consumer_col,
            "debit_txn_count",
            "outlying_spending_count",
            "outlying_spending_ratio"
        ])

    g = df.groupby(consumer_col)

    # Compute IQR stats per consumer
    q1 = g[amt_col].quantile(0.25)
    q3 = g[amt_col].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + iqr_k * iqr

    thresholds = upper_bound.rename("upper_bound").reset_index()

    # Merge thresholds back
    df = df.merge(thresholds, on=consumer_col, how="left")

    # Flag outliers
    df["is_outlier"] = df[amt_col] > df["upper_bound"]

    # Aggregate counts
    result = (
        df.groupby(consumer_col)
          .agg(
              debit_txn_count=(amt_col, "size"),
              outlying_spending_count=("is_outlier", "sum")
          )
          .reset_index()
    )

    # Ratio
    result["outlying_spending_ratio"] = (
        result["outlying_spending_count"] / result["debit_txn_count"]
    )

    return result