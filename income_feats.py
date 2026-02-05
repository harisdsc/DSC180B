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

    income_features = (
        total_income
        .merge(avg_monthly_income, on="prism_consumer_id", how="left")
        .merge(income_consistency, on="prism_consumer_id", how="left")
        .merge(income_sources, on="prism_consumer_id", how="left")
        .merge(
            benefits_flag[["prism_consumer_id", "has_benefits_income"]],
            on="prism_consumer_id",
            how="left"
        )
    )

    return income_features