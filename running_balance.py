import pandas as pd


# ======================
# Configuration
# ======================
CONS_PATH = "/uss/hdsi-prismdata/q2-ucsd-consDF.pqt"
ACCT_PATH = "/uss/hdsi-prismdata/q2-ucsd-acctDF.pqt"
TRXN_PATH = "/uss/hdsi-prismdata/q2-ucsd-trxnDF.pqt"


# ======================
# Load data
# ======================
def load_data():
    consDF = pd.read_parquet(CONS_PATH)
    acctDF = pd.read_parquet(ACCT_PATH)
    trxnDF = pd.read_parquet(TRXN_PATH)

    consDF = consDF.drop(columns=["credit_score"]).dropna()
    acctDF = acctDF[acctDF["prism_consumer_id"].isin(consDF['prism_consumer_id'])]
    acctDF["balance_date"] = pd.to_datetime(acctDF["balance_date"])
    trxnDF = trxnDF[trxnDF["prism_consumer_id"].isin(consDF['prism_consumer_id'])]
    trxnDF = trxnDF.drop_duplicates()
    trxnDF["posted_date"] = pd.to_datetime(trxnDF["posted_date"])

    return consDF, acctDF, trxnDF


# ======================
# Compute first snapshot
# ======================
def compute_first_snapshot(acctDF):
    acct_consumer = (
        acctDF
        .groupby(["prism_consumer_id", "balance_date"], as_index=False)["balance"]
        .sum()
    )

    first_snapshot = (
        acct_consumer
        .sort_values("balance_date")
        .groupby("prism_consumer_id", as_index=False)
        .first()
        .rename(columns={
            "balance_date": "start_date",
            "balance": "start_balance"
        })
    )

    return first_snapshot


# ======================
# Prepare transactions
# ======================
def prepare_transactions(trxnDF):
    trxnDF = trxnDF.copy()
    trxnDF["signed_amount"] = trxnDF["amount"].where(
        trxnDF["credit_or_debit"] == "CREDIT",
        -trxnDF["amount"]
    )
    return trxnDF


# ======================
# Forward balances
# ======================
def compute_forward_balances(trxnDF, first_snapshot):
    trxn_with_start = trxnDF.merge(
        first_snapshot,
        on="prism_consumer_id",
        how="inner"
    )

    trxn_with_start = trxn_with_start[
        trxn_with_start["posted_date"] >= trxn_with_start["start_date"]
    ].sort_values(["prism_consumer_id", "posted_date"])

    trxn_with_start["cum_trxn"] = (
        trxn_with_start
        .groupby("prism_consumer_id")["signed_amount"]
        .cumsum()
    )

    trxn_with_start["running_balance"] = (
        trxn_with_start["start_balance"] + trxn_with_start["cum_trxn"]
    )

    start_rows = first_snapshot.rename(columns={
        "start_date": "posted_date",
        "start_balance": "running_balance"
    })
    start_rows["signed_amount"] = 0.0

    cols = [
    "prism_consumer_id",
    "posted_date",
    "signed_amount",
    "running_balance",
    "category",
    "credit_or_debit"
    ]

    start_rows = start_rows.reindex(columns=cols)
    trxn_with_start = trxn_with_start.reindex(columns=cols)


    forward_df = pd.concat([
        start_rows[["prism_consumer_id", "posted_date", "signed_amount", "running_balance", "category",
    "credit_or_debit"]],
        trxn_with_start[["prism_consumer_id", "posted_date", "signed_amount", "running_balance", "category",
    "credit_or_debit"]],
    ]).sort_values(["prism_consumer_id", "posted_date"])

    return forward_df, trxn_with_start


# ======================
# Backward balances
# ======================
def compute_backward_balances(trxnDF, first_snapshot):
    trxn_before = trxnDF.merge(
        first_snapshot,
        on="prism_consumer_id",
        how="inner"
    )

    trxn_before = trxn_before[
        trxn_before["posted_date"] < trxn_before["start_date"]
    ].sort_values(
        ["prism_consumer_id", "posted_date"],
        ascending=[True, False]
    )

    trxn_before["cum_trxn_back"] = (
        trxn_before
        .groupby("prism_consumer_id")["signed_amount"]
        .cumsum()
    )

    trxn_before["running_balance"] = (
        trxn_before["start_balance"] - trxn_before["cum_trxn_back"]
    )

    start_rows_back = first_snapshot.rename(columns={
        "start_date": "posted_date",
        "start_balance": "running_balance"
    })
    start_rows_back["signed_amount"] = 0.0
    cols = [
    "prism_consumer_id",
    "posted_date",
    "signed_amount",
    "running_balance",
    "category",
    "credit_or_debit"
    ]
    start_rows_back = start_rows_back.reindex(columns=cols)
    trxn_before = trxn_before.reindex(columns=cols)
    
    backward_df = pd.concat([
        trxn_before[["prism_consumer_id", "posted_date", "signed_amount", "running_balance", "category",
    "credit_or_debit"]],
        start_rows_back[["prism_consumer_id", "posted_date", "signed_amount", "running_balance", "category",
    "credit_or_debit"]],
    ]).sort_values(["prism_consumer_id", "posted_date"])

    return backward_df, trxn_before


# ======================
# Full balance history
# ======================
def build_full_balance_df(trxn_before, trxn_with_start, first_snapshot):
    snapshot_row = first_snapshot.rename(columns={
        "start_date": "posted_date",
        "start_balance": "running_balance"
    })
    snapshot_row["signed_amount"] = 0.0
    cols = [
    "prism_consumer_id",
    "posted_date",
    "signed_amount",
    "running_balance",
    "category",
    "credit_or_debit"
    ]
    snapshot_row = snapshot_row.reindex(columns=cols)
    full_balance_df = (
        pd.concat([
            trxn_before[["prism_consumer_id", "posted_date", "signed_amount", "running_balance", "category",
    "credit_or_debit"]],
            snapshot_row[["prism_consumer_id", "posted_date", "signed_amount", "running_balance", "category",
    "credit_or_debit"]],
            trxn_with_start[["prism_consumer_id", "posted_date", "signed_amount", "running_balance", "category",
    "credit_or_debit"]],
        ])
        .sort_values(["prism_consumer_id", "posted_date"])
        .reset_index(drop=True)
    )

    return full_balance_df


# ======================
# Main
# ======================
def main():
    consDF, acctDF, trxnDF = load_data()
    first_snapshot = compute_first_snapshot(acctDF)
    trxnDF = prepare_transactions(trxnDF)

    forward_df, trxn_with_start = compute_forward_balances(trxnDF, first_snapshot)
    backward_df, trxn_before = compute_backward_balances(trxnDF, first_snapshot)
    full_balance_df = build_full_balance_df(trxn_before, trxn_with_start, first_snapshot)

    # Example: save outputs
    forward_df.to_parquet("forward_running_balance.pqt", index=False)
    backward_df.to_parquet("backward_running_balance.pqt", index=False)
    full_balance_df.to_parquet("full_running_balance.pqt", index=False)


if __name__ == "__main__":
    main()