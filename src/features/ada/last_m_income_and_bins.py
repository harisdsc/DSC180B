import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import OneHotEncoder

def compute_income_from_eval_windows(
    txn_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    income_cats = [2,3,5,7,8,9,49],
    windows=[1,3,6,12],  # months
    consumer_col: str = "prism_consumer_id",
    txn_date_col: str = "posted_date",
    eval_date_col: str = "evaluation_date",
    amt_col: str = "amount",
    direction_col: str = "credit_or_debit",
    category_col: str = "category",
) -> pd.DataFrame:
    """
    For each consumer, compute income over multiple month windows
    prior to (and including) evaluation_date.

    Returns:
        DataFrame with one row per consumer and columns:
        income_last_{W}m for each W in windows
    """

    tx = txn_df.copy()
    cons = cons_df[[consumer_col, eval_date_col]].copy()

    tx[txn_date_col] = pd.to_datetime(tx[txn_date_col])
    cons[eval_date_col] = pd.to_datetime(cons[eval_date_col])

    # Keep only income transactions
    tx = tx[
        (tx[direction_col] == "CREDIT") &
        (tx[category_col].isin(income_cats))
    ].copy()

    # Attach evaluation_date
    tx = tx.merge(cons, on=consumer_col, how="inner")

    # Start with full consumer base
    output = cons[[consumer_col]].drop_duplicates().copy()

    for W in windows:

        start = tx[eval_date_col] - pd.DateOffset(months=W)

        tx_window = tx[
            (tx[txn_date_col] >= start) &
            (tx[txn_date_col] <= tx[eval_date_col])
        ]

        agg = (
            tx_window
            .groupby(consumer_col, as_index=False)[amt_col]
            .sum()
            .rename(columns={amt_col: f"income_last_{W}m"})
        )

        output = output.merge(agg, on=consumer_col, how="left")

    # Fill consumers with no income as 0
    for W in windows:
        col = f"income_last_{W}m"
        if col in output.columns:
            output[col] = output[col].fillna(0.0)

    return output


def fit_trans_income_binner_and_ohe(income_df, q=5, prefix="income_last_"):
    """
    TRAINING:
      - For each income window column:
          try qcut -> store bin_edges + scheme
          fallback to cut -> store bin_edges + scheme
      - Create *_bin columns
      - Fit OneHotEncoder on *_bin
    Returns:
      out_df, artifacts
        artifacts = {
          "binning": {col: {"edges": array, "method": "qcut"/"cut", "labels": [...]}}
          "bin_cols": [...],
          "ohe": fitted OneHotEncoder
        }
    """
    df = income_df.copy()
    income_cols = [c for c in df.columns if c.startswith(prefix)]

    binning = {}
    bin_cols = []

    for col in income_cols:
        bin_col = f"{col}_bin"
        bin_cols.append(bin_col)

        # If you want to treat NaNs explicitly, keep them as NaN here.
        # OHE can handle unknowns, but NaN will be a category unless you impute or set dtype.
        x = df[col]

        try:
            labels = [f"Q{i+1}" for i in range(q)]
            binned, edges = pd.qcut(
                x, q=q, labels=labels, duplicates="drop", retbins=True
            )
            method = "qcut"
            # When duplicates="drop", actual number of bins may be < q:
            actual_bins = len(edges) - 1
            labels = labels[:actual_bins]
            # Re-run with the truncated labels to keep things consistent
            binned = pd.cut(x, bins=edges, labels=labels, include_lowest=True)

        except ValueError:
            labels = [f"B{i+1}" for i in range(q)]
            binned, edges = pd.cut(
                x, bins=q, labels=labels, retbins=True, include_lowest=True
            )
            method = "cut"

        df[bin_col] = binned.astype("object")  # keep as categorical-like
        binning[col] = {"edges": np.asarray(edges), "method": method, "labels": list(labels)}

    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    transformed = ohe.fit_transform(df[bin_cols])

    one_hot_df = pd.DataFrame(
        transformed,
        columns=ohe.get_feature_names_out(bin_cols),
        index=df.index
    )

    out = pd.concat([df, one_hot_df], axis=1)

    artifacts = {"binning": binning, "bin_cols": bin_cols, "ohe": ohe}
    return out, artifacts


def transform_income_with_artifacts(income_df, artifacts, prefix="income_last_"):
    """
    INFERENCE:
      - Use stored bin edges per column to create *_bin
      - Use stored fitted OHE to transform
      - Returns out_df with same one-hot columns as training
    """
    df = income_df.copy()
    binning = artifacts["binning"]
    bin_cols = artifacts["bin_cols"]
    ohe = artifacts["ohe"]

    # Build bins deterministically using stored edges
    for col, meta in binning.items():
        edges = meta["edges"].copy()
        labels = meta["labels"]

        # Optional: make bins cover any future out-of-range values
        # (prevents NaNs when new data exceeds training min/max)
        edges[0] = -np.inf
        edges[-1] = np.inf

        bin_col = f"{col}_bin"
        df[bin_col] = pd.cut(
            df[col],
            bins=edges,
            labels=labels,
            include_lowest=True
        ).astype("object")

    # Ensure all expected bin cols exist (in case some income columns are missing)
    for bc in bin_cols:
        if bc not in df.columns:
            df[bc] = np.nan

    transformed = ohe.transform(df[bin_cols])
    one_hot_df = pd.DataFrame(
        transformed,
        columns=ohe.get_feature_names_out(bin_cols),
        index=df.index
    )
    out = pd.concat([df, one_hot_df], axis=1)
    return out