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

    account_features = (
        agg_bal
        .merge(pivot_bal, on="prism_consumer_id", how="left")
        .merge(liquidity_flags, on="prism_consumer_id", how="left")
    )

    return account_features

