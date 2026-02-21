import pandas as pd
from datetime import date
from sklearn.preprocessing import OneHotEncoder

def compute_income_from_eval_windows(
    txn_df,
    cons_df,
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

    #for new data, no eval_date col so use today's date?
    if eval_date_col in cons_df.columns:
        cons_df[eval_date_col] = pd.to_datetime(cons_df[eval_date_col], errors="coerce")
    else:
        today = pd.Timestamp(date.today())
        cons_df[eval_date_col] = today
        
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


def bin_income_quantiles(income_df, q=5, prefix="income_last_"): 
    """ 
    Apply quantile binning to all income window columns. 
    
    Example expected columns: 
    income_last_1m 
    income_last_3m 
    income_last_6m ... 
    """ 
    
    df = income_df.copy() 

    # Find all income window columns 
    income_cols = [c for c in df.columns if c.startswith(prefix)] 
    
    for col in income_cols: 
        try: df[f"{col}_bin"] = pd.qcut( 
                df[col], 
                q=q, 
                labels=[f"Q{i+1}" for i in range(q)], 
                duplicates="drop" # prevents crash if too many identical values 
            ) 
        except ValueError: 
            # If too few unique values, fallback to equal-width bins 
            df[f"{col}_bin"] = pd.cut( 
                df[col], 
                bins=q, 
                labels=[f"B{i+1}" for i in range(q)] 
            ) 
                
    return df