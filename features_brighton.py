"""
consumer_features.py

End-to-end feature engineering pipeline for PRISM consumer-level modeling.
"""

import pandas as pd
import numpy as np


# ======================
# Paths
# ======================
CONS_PATH = "/uss/hdsi-prismdata/q2-ucsd-consDF.pqt"
ACCT_PATH = "/uss/hdsi-prismdata/q2-ucsd-acctDF.pqt"
TRXN_PATH = "/uss/hdsi-prismdata/q2-ucsd-trxnDF.pqt"
CATMAP_PATH = "/uss/hdsi-prismdata/q2-ucsd-cat-map.csv"
FULL_BAL_PATH = "full_running_balance.pqt"


# ======================
# Load data
# ======================
def load_data():
    consDF = (
        pd.read_parquet(CONS_PATH)
        .drop(columns=["credit_score"])
        .dropna()
    )
    acctDF = pd.read_parquet(ACCT_PATH)
    trxnDF = pd.read_parquet(TRXN_PATH)
    catmap = pd.read_csv(CATMAP_PATH)

    trxnDF["posted_date"] = pd.to_datetime(trxnDF["posted_date"])
    return consDF, acctDF, trxnDF, catmap

def load_full_balance_df():
    full_balance_df = pd.read_parquet(FULL_BAL_PATH)
    full_balance_df["posted_date"] = pd.to_datetime(full_balance_df["posted_date"])
    return full_balance_df


# ======================
# Account balance features
# ======================
def build_account_features(acctDF):
    agg_bal = (
        acctDF
        .groupby("prism_consumer_id", as_index=False)
        .agg(
            total_balance=("balance", "sum"),
            avg_balance=("balance", "mean"),
            max_balance=("balance", "max"),
            min_balance=("balance", "min"),
            std_balance=("balance", "std"),
            num_accounts=("balance", "count"),
        )
    )

    pivot_bal = (
        acctDF
        .pivot_table(
            index="prism_consumer_id",
            columns="account_type",
            values="balance",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    pivot_bal["checking_to_savings_ratio"] = (
        pivot_bal.get("CHECKING", 0) / (pivot_bal.get("SAVINGS", 0) + 1)
    )
    pivot_bal["has_savings_account"] = (
        pivot_bal.get("SAVINGS", 0) > 0
    ).astype(int)

    pivot_bal = pivot_bal[
        [
            "prism_consumer_id",
            "CHECKING",
            "SAVINGS",
            "checking_to_savings_ratio",
            "has_savings_account",
        ]
    ]

    acctDF["liquid_balance"] = (
        acctDF
        .assign(
            liquid_component=lambda df: np.where(
                df["account_type"].isin(["CHECKING", "SAVINGS"]),
                df["balance"],
                0
            )
        )
        .groupby("prism_consumer_id")["liquid_component"]
        .transform("sum")
    )

    liquidity_flags = (
        acctDF
        .groupby("prism_consumer_id", as_index=False)["liquid_balance"]
        .first()
    )
    liquidity_flags["low_liquidity"] = (liquidity_flags["liquid_balance"] < 500).astype(int)
    liquidity_flags["is_liquid"] = (liquidity_flags["liquid_balance"] > 0).astype(int)

    return agg_bal, pivot_bal, liquidity_flags


# ======================
# Income features
# ======================
def build_income_features(trxnDF, catmap):
    trxnDF = trxnDF.merge(
        catmap,
        left_on="category",
        right_on="category_id",
        how="left"
    )

    INCOME_CATEGORIES = [
        "DEPOSIT",
        "PAYCHECK",
        "PAYCHECK_PLACEHOLDER",
        "INVESTMENT_INCOME",
        "OTHER_BENEFITS",
        "UNEMPLOYMENT_BENEFITS",
        "PENSION",
    ]

    income_txn = trxnDF[
        (trxnDF["credit_or_debit"] == "CREDIT") &
        (trxnDF["category_y"].isin(INCOME_CATEGORIES))
    ].copy()

    income_txn["month"] = income_txn["posted_date"].dt.to_period("M")

    total_income = (
        income_txn
        .groupby("prism_consumer_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "total_income"})
    )

    avg_monthly_income = (
        income_txn
        .groupby(["prism_consumer_id", "month"])["amount"]
        .sum()
        .groupby("prism_consumer_id")
        .mean()
        .reset_index(name="avg_monthly_income")
    )

    income_consistency = (
        income_txn
        .groupby("prism_consumer_id")["month"]
        .nunique()
        .reset_index(name="num_income_months")
    )

    income_sources = (
        income_txn
        .groupby("prism_consumer_id")["category_y"]
        .nunique()
        .reset_index(name="num_income_sources")
    )

    benefits_flag = (
        income_txn[income_txn["category_x"].isin([8, 9, 49])]
        .groupby("prism_consumer_id")
        .size()
        .reset_index(name="benefits_txn_count")
    )
    benefits_flag["has_benefits_income"] = (
        benefits_flag["benefits_txn_count"] > 0
    ).astype(int)

    return (
        total_income,
        avg_monthly_income,
        income_consistency,
        income_sources,
        benefits_flag[["prism_consumer_id", "has_benefits_income"]],
    )


# ======================
# Balance dynamics (from full_balance_df)
# ======================
def build_balance_dynamics(full_balance_df):
    full_balance_df = full_balance_df.sort_values(
        ["prism_consumer_id", "posted_date"]
    )

    full_balance_df["daily_balance_change"] = (
        full_balance_df
        .groupby("prism_consumer_id")["running_balance"]
        .diff()
    )

    daily_change_features = (
        full_balance_df
        .groupby("prism_consumer_id", as_index=False)
        .agg(
            mean_daily_change=("daily_balance_change", "mean"),
            std_daily_change=("daily_balance_change", "std"),
            max_daily_drop=("daily_balance_change", "min"),
            max_daily_increase=("daily_balance_change", "max"),
        )
    )

    latest_dates = (
        full_balance_df
        .groupby("prism_consumer_id")["posted_date"]
        .max()
        .reset_index(name="latest_date")
    )

    full_balance_df = full_balance_df.merge(latest_dates, on="prism_consumer_id")
    full_balance_df["date_30d"] = full_balance_df["latest_date"] - pd.Timedelta(days=30)
    full_balance_df["date_90d"] = full_balance_df["latest_date"] - pd.Timedelta(days=90)
    full_balance_df["date_365d"] = full_balance_df["latest_date"] - pd.Timedelta(days=365)

    def last_balance_before(df, date_col):
        return (
            df[df["posted_date"] <= df[date_col]]
            .groupby("prism_consumer_id")["running_balance"]
            .last()
        )

    ending_balance = (
        full_balance_df
        .groupby("prism_consumer_id")["running_balance"]
        .last()
    )

    recent_balance_features = pd.DataFrame({
        "recent_30d_balance_change": ending_balance - last_balance_before(full_balance_df, "date_30d"),
        "recent_90d_balance_change": ending_balance - last_balance_before(full_balance_df, "date_90d"),
    }).fillna(0).reset_index()

    mean_balance_features = (
        pd.concat(
            [
                full_balance_df[full_balance_df["posted_date"] >= full_balance_df["date_30d"]]
                .groupby("prism_consumer_id")["running_balance"]
                .mean()
                .rename("mean_balance_30d"),

                full_balance_df[full_balance_df["posted_date"] >= full_balance_df["date_90d"]]
                .groupby("prism_consumer_id")["running_balance"]
                .mean()
                .rename("mean_balance_90d"),

                full_balance_df[full_balance_df["posted_date"] >= full_balance_df["date_365d"]]
                .groupby("prism_consumer_id")["running_balance"]
                .mean()
                .rename("mean_balance_365d"),
            ],
            axis=1
        )
        .reset_index()
    )

    return daily_change_features, recent_balance_features, mean_balance_features


# ======================
# Main
# ======================
def main(full_balance_df):
    consDF, acctDF, trxnDF, catmap = load_data()
    full_balance_df = load_full_balance_df()

    agg_bal, pivot_bal, liquidity_flags = build_account_features(acctDF)
    income_feats = build_income_features(trxnDF, catmap)
    balance_feats = build_balance_dynamics(full_balance_df)

    features = consDF[["prism_consumer_id"]]

    for df in (
        agg_bal,
        pivot_bal,
        liquidity_flags,
        *income_feats,
        *balance_feats,
    ):
        features = features.merge(df, on="prism_consumer_id", how="left")

    features = features.fillna(0)

    features.to_parquet("consumer_features.pqt", index=False)
    return features


if __name__ == "__main__":
    # full_balance_df must already be constructed before running
    raise RuntimeError(
        "Pass full_balance_df to main() after computing running balances."
    )