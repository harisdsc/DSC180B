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

     balance_dynamics_features = (
        daily_change_features
        .merge(recent_balance_features, on="prism_consumer_id", how="left")
        .merge(mean_balance_features, on="prism_consumer_id", how="left")
    )

    return balance_dynamics_features
