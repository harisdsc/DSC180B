import pandas as pd
import numpy as np
from load_data import load_data

def consumer_agg(
    df: pd.DataFrame,
    group_col: str,
    feat_cols: list[str],
    window,
    stats=("mean", "median", "std", "min", "max"),
    prefix_map=None,
    label: str | None = None,
) -> pd.DataFrame:
    """
    Consumer-level aggregation.

    Two modes:

    A) Multi-feature mode (default):
       feat_cols = ["x_3m", "y_3m"]  ->
         avg_x_3m, med_x_3m, sd_x_3m, min_x_3m, max_x_3m,
         avg_y_3m, ...

    B) Single-feature pretty naming:
       feat_cols = ["net_flow_6m"], window=6, label="netflow" ->
         avg_6m_netflow, med_6m_netflow, sd_6m_netflow, min_6m_netflow, max_6m_netflow
    """
    if prefix_map is None:
        prefix_map = {"mean": "avg", "median": "med", "std": "sd", "min": "min", "max": "max"}

    feat_cols = list(feat_cols)

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"consumer_agg: missing columns in df: {missing}")

    agg_spec = {}

    single_pretty = (window is not None) and (label is not None) and (len(feat_cols) == 1)
    if single_pretty:
        c = feat_cols[0]
        for s in stats:
            agg_spec[f"{prefix_map.get(s, s)}_{int(window)}m_{label}"] = (c, s)
    else:
        for c in feat_cols:
            for s in stats:
                agg_spec[f"{prefix_map.get(s, s)}_{c}"] = (c, s)

    return df.groupby(group_col, as_index=False).agg(**agg_spec)


def build_monthly_cashflows(
    txn_df: pd.DataFrame,
    window: int,
    income_cats,
    date_col="posted_date",
    amt_col="amount",
    consumer_col="prism_consumer_id",
    cd_col="credit_or_debit",
    cat_col="category",
    fill_months=True,
    min_periods=None,
    return_consumer_level=False,
):
    """
    Builds monthly income/spend and rolling net flow features.

    Fixes:
      - avoids groupby.apply DeprecationWarning via include_groups=False
      - avoids risky reindex(fill_value=0.0) by only filling numeric columns
      - normalizes CREDIT/DEBIT casing
    """
    w = int(window)
    if min_periods is None:
        min_periods = w

    df = txn_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, consumer_col])
    df = df.sort_values([consumer_col, date_col])

    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    # normalize CREDIT/DEBIT
    dir_upper = df[cd_col].astype(str).str.upper()

    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
    df = df.dropna(subset=[amt_col])

    df["income_amt"] = np.where(
        (dir_upper == "CREDIT") & (df[cat_col].isin(income_cats)),
        df[amt_col].astype(float),
        0.0,
    )
    df["spend_amt"] = np.where(
        (dir_upper == "DEBIT"),
        df[amt_col].astype(float),
        0.0,
    )

    monthly = (
        df.groupby([consumer_col, "month"], as_index=False)
          .agg(income=("income_amt", "sum"),
               spend=("spend_amt", "sum"))
          .sort_values([consumer_col, "month"])
          .reset_index(drop=True)
    )

    # --- fill missing months within each consumer (safe fill) ---
    if fill_months and not monthly.empty:
        fill_cols = ["income", "spend"]
    
        def _fill(g: pd.DataFrame) -> pd.DataFrame:
            cid = g.name  # <-- group key (consumer id) when include_groups=False
            full = pd.date_range(g["month"].min(), g["month"].max(), freq="MS")
    
            g2 = (
                g.set_index("month")
                 .reindex(full)
                 .rename_axis("month")
                 .reset_index()
            )
    
            g2[consumer_col] = cid
    
            for c in fill_cols:
                if c not in g2.columns:
                    g2[c] = 0.0
            g2[fill_cols] = g2[fill_cols].fillna(0.0)
    
            return g2
    
        monthly = (
            monthly
            .groupby(consumer_col, group_keys=False)
            .apply(_fill, include_groups=False)
            .sort_values([consumer_col, "month"])
            .reset_index(drop=True)
        )


    # --- rolling features ---
    g = monthly.groupby(consumer_col, group_keys=False)

    if w == 1:
        monthly["income_1m"] = monthly["income"]
        monthly["spend_1m"] = monthly["spend"]
        monthly["net_flow_1m"] = monthly["income_1m"] - monthly["spend_1m"]
        feat_col = "net_flow_1m"
    else:
        monthly[f"income_{w}m"] = g["income"].transform(lambda s: s.rolling(w, min_periods=min_periods).sum())
        monthly[f"spend_{w}m"]  = g["spend"].transform(lambda s: s.rolling(w, min_periods=min_periods).sum())
        monthly[f"net_flow_{w}m"] = monthly[f"income_{w}m"] - monthly[f"spend_{w}m"]
        feat_col = f"net_flow_{w}m"

    if not return_consumer_level:
        return monthly

    return consumer_agg(
        df=monthly,
        group_col=consumer_col,
        feat_cols=[feat_col],
        window=w,
        label="netflow",
    )

def build_monthly_category_to_income(
    txn_df: pd.DataFrame,
    income_cats,
    window,
    category_ids,
    date_col: str = "posted_date",
    consumer_col: str = "prism_consumer_id",
    amt_col: str = "amount",
    direction_col: str = "credit_or_debit",
    category_col: str = "category",
    fill_missing_months: bool = True,
    min_periods: int | None = None,
    consumer_level: bool = False,
    agg_stats=("mean", "median", "std", "min", "max"),
) -> pd.DataFrame:

    # --- normalize category_ids ---
    if isinstance(category_ids, (int, np.integer)):
        category_ids = [int(category_ids)]
    else:
        category_ids = [int(x) for x in category_ids]
    category_ids = sorted(set(category_ids))

    # --- prep txns ---
    t = txn_df[[date_col, consumer_col, amt_col, direction_col, category_col]].copy()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t = t.dropna(subset=[date_col, consumer_col])

    t[amt_col] = pd.to_numeric(t[amt_col], errors="coerce")
    t = t.dropna(subset=[amt_col])

    t["month"] = t[date_col].dt.to_period("M").dt.to_timestamp()

    dir_upper = t[direction_col].astype(str).str.upper()
    is_credit = dir_upper.eq("CREDIT")
    is_debit  = dir_upper.eq("DEBIT")

    income_cats_set = set(int(x) for x in income_cats)

    # --- monthly income ---
    income_m = (
        t.loc[is_credit & t[category_col].isin(income_cats_set)]
         .groupby([consumer_col, "month"], as_index=False)[amt_col]
         .sum()
         .rename(columns={amt_col: "income_1m"})
    )

    # --- monthly spend for requested categories (all at once) ---
    spend_1m_cols = [f"cat{cid}_spend_1m" for cid in category_ids]

    spend_m = (
        t.loc[is_debit & t[category_col].isin(category_ids)]
         .groupby([consumer_col, "month", category_col])[amt_col]
         .sum()
         .unstack(category_col)
    )

    if not spend_m.empty:
        spend_m = spend_m.reindex(columns=category_ids)  # stable order + include missing cats
        spend_m.columns = spend_1m_cols
        spend_m = spend_m.reset_index()
    else:
        spend_m = pd.DataFrame(columns=[consumer_col, "month"] + spend_1m_cols)

    # --- base (one merge) ---
    base = income_m.merge(spend_m, on=[consumer_col, "month"], how="outer")

    if base.empty:
        cols = [consumer_col, "month", "income_1m"] + spend_1m_cols
        return pd.DataFrame(columns=cols)

    base = base.sort_values([consumer_col, "month"]).reset_index(drop=True)

    # fill missing monthly values
    base["income_1m"] = base["income_1m"].fillna(0.0)
    for c in spend_1m_cols:
        if c not in base.columns:
            base[c] = 0.0
    base[spend_1m_cols] = base[spend_1m_cols].fillna(0.0)

    # --- fill missing months within each consumer (no .name / no include_groups needed) ---
    if fill_missing_months and not base.empty:
        fill_cols = ["income_1m"] + spend_1m_cols

        spans = (
            base.groupby(consumer_col, as_index=False)["month"]
                .agg(min_month="min", max_month="max")
        )
        spans["month"] = spans.apply(
            lambda r: pd.date_range(r["min_month"], r["max_month"], freq="MS"),
            axis=1
        )
        grid = spans[[consumer_col, "month"]].explode("month", ignore_index=True)

        base = grid.merge(base, on=[consumer_col, "month"], how="left")

        for c in fill_cols:
            if c not in base.columns:
                base[c] = 0.0
        base[fill_cols] = base[fill_cols].fillna(0.0)

        base = base.sort_values([consumer_col, "month"]).reset_index(drop=True)

    # --- rolling features ---
    gb = base.groupby(consumer_col, group_keys=False)
    frames = []

    # (A) If window includes 1: only create ratio_1m (DON'T re-add income_1m/spend_1m)
    if 1 in set(int(w) for w in window):
        denom_1m = base["income_1m"].replace(0, np.nan)
        ratio_1m = base[spend_1m_cols].div(denom_1m, axis=0).rename(
            columns=lambda c: c.replace("_spend_1m", "_to_income_ratio_1m")
        )
        frames.append(ratio_1m)

    # (B) W > 1: create income_Wm, spend_Wm, ratio_Wm
    for W in [int(w) for w in window if int(w) != 1]:
        mp = W if min_periods is None else int(min_periods)

        income_roll = gb["income_1m"].transform(lambda s: s.rolling(W, min_periods=mp).sum())
        denom = income_roll.replace(0, np.nan)

        spend_roll = gb[spend_1m_cols].transform(lambda df: df.rolling(W, min_periods=mp).sum())
        spend_roll = spend_roll.rename(columns=lambda c: c.replace("_spend_1m", f"_spend_{W}m"))

        ratio = spend_roll.div(denom, axis=0).rename(
            columns=lambda c: c.replace("_spend_", "_to_income_ratio_")
        )

        frames.append(pd.concat([income_roll.rename(f"income_{W}m"), spend_roll, ratio], axis=1))

    base = pd.concat([base] + frames, axis=1).copy()

    # --- return monthly or consumer-level ---
    if not consumer_level:
        return base

    agg_cols = []
    for cat_id in category_ids:
        label = f"cat{cat_id}"
        for W in [int(w) for w in window]:
            agg_cols.extend([
                f"{label}_spend_{W}m" if W != 1 else f"{label}_spend_1m",
                f"{label}_to_income_ratio_{W}m",
            ])

    return consumer_agg(
        df=base,
        group_col=consumer_col,
        feat_cols=agg_cols,
        window=window,
        stats=agg_stats,
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    consDF, testDF, acctDF, trxnDF, cat_map = load_data()
    
    income_cats = [2,3,5,7,8,9,49]
    windows = [1,3,6,9]
    categories = cat_map['category_id'].unique()

    m1 = build_monthly_cashflows(
    trxnDF,
    window=1,
    income_cats=income_cats,
    return_consumer_level=False
)
    netflow_feats = (
        m1.groupby("prism_consumer_id", as_index=False)
          .agg(
              months_observed=("month", "nunique"),
              total_income=("income_1m", "sum"),
              total_spend=("spend_1m", "sum"),
              total_net_flow=("net_flow_1m", "sum"),
          )
    )
    
    # add windowed consumer-level summary stats --> 1m, 3m, 6m, etc.
    for w in windows:
        cons_w = build_monthly_cashflows(
            trxnDF,
            window=w,
            income_cats=income_cats,
            return_consumer_level=True
        )
        netflow_feats = netflow_feats.merge(cons_w, on="prism_consumer_id", how="left")

    cat_ratio_feats = build_monthly_category_to_income(
        txn_df=trxnDF,
        income_cats=income_cats,
        window=windows,
        category_ids=categories,
        consumer_level=True,
    )

    mean_impute = netflow_feats.merge(cat_ratio_feats, on = 'prism_consumer_id')

    for c in mean_impute.columns:
        if c != 'prism_consumer_id':
            mean_impute[c] = mean_impute[c].fillna(mean_impute[c].mean())
    
    # Example: save outputs
    return mean_impute
    # .to_parquet("income_ratios.pqt", index=False)

if __name__ == "__main__":
    features = main()