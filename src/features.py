import pandas as pd
import numpy as np
import pywt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_running_balance(t_df, a_df):
    t_df = t_df.copy()
    t_df['signed_amount'] = np.where(t_df['credit_or_debit'] == 'DEBIT', -t_df['amount'], t_df['amount'])
    sort_cols = ['prism_consumer_id', 'posted_date']
    if 'prism_transaction_id' in t_df.columns: sort_cols.append('prism_transaction_id')
    t_df = t_df.sort_values(sort_cols)
    t_df['raw_cumsum'] = t_df.groupby('prism_consumer_id')['signed_amount'].cumsum()
    snapshot = a_df.sort_values('balance_date').groupby('prism_consumer_id').tail(1)
    snapshot = snapshot.rename(columns={'balance': 'snapshot_balance', 'balance_date': 'snapshot_date'})
    df_merged = t_df.merge(snapshot[['prism_consumer_id', 'snapshot_date', 'snapshot_balance']], on='prism_consumer_id', how='left')
    anchor = df_merged[df_merged['posted_date'] <= df_merged['snapshot_date']].groupby('prism_consumer_id')['raw_cumsum'].last().rename('cumsum_at_snapshot')
    df_merged = df_merged.merge(anchor, on='prism_consumer_id', how='left')
    df_merged['running_balance'] = (df_merged['raw_cumsum'] - df_merged['cumsum_at_snapshot']) + df_merged['snapshot_balance']
    out_cols = ['prism_consumer_id', 'posted_date', 'category_id', 'category', 'amount', 'signed_amount', 'credit_or_debit', 'running_balance', 'DQ_TARGET']
    return df_merged[[c for c in out_cols if c in df_merged.columns]]

def extract_consumer_habits(df):
    df = df.copy()
    df['posted_date'] = pd.to_datetime(df['posted_date'])
    df['day_name'] = df['posted_date'].dt.day_name().str.lower()
    df['day_of_month'] = df['posted_date'].dt.day
    day_avg = df.groupby(['prism_consumer_id', 'category', 'day_name'])['signed_amount'].mean().reset_index()
    weekly_pivot = day_avg.pivot_table(index='prism_consumer_id', columns=['category', 'day_name'], values='signed_amount').fillna(0)
    weekly_pivot.columns = [f"cat_{c}_{d}" for c, d in weekly_pivot.columns]
    dom_avg = df.groupby(['prism_consumer_id', 'category', 'day_of_month'])['signed_amount'].mean().reset_index()
    monthly_pivot = dom_avg.pivot_table(index='prism_consumer_id', columns=['category', 'day_of_month'], values='signed_amount').fillna(0)
    monthly_pivot.columns = [f"cat_{c}_dom_{d}" for c, d in monthly_pivot.columns]

    def get_wavelet_power(series):
        if len(series) < 14 or series.sum() == 0: return 0, 0
        vals = series.values - series.mean()
        cf = pywt.central_frequency('morl')
        if len(vals) >= 28:
            coef, _ = pywt.cwt(vals, [cf*7, cf*30], 'morl')
            return np.mean(np.abs(coef[0])**2), np.mean(np.abs(coef[1])**2)
        coef, _ = pywt.cwt(vals, [cf*7], 'morl')
        return np.mean(np.abs(coef[0])**2), 0

    cwt_results = []
    for (cons_id, cat), group in df.groupby(['prism_consumer_id', 'category']):
        daily = group.groupby('posted_date')['signed_amount'].sum()
        series = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
        w, m = get_wavelet_power(series)
        cwt_results.append({'prism_consumer_id': cons_id, 'category': cat, 'cwt_weekly_habit': w, 'cwt_monthly_habit': m})
    
    cwt_df = pd.DataFrame(cwt_results)
    if not cwt_df.empty:
        cwt_pivot_w = cwt_df.pivot(index='prism_consumer_id', columns='category', values='cwt_weekly_habit').fillna(0)
        cwt_pivot_w.columns = [f"cat_{c}_cwt_weekly" for c in cwt_pivot_w.columns]
        cwt_pivot_m = cwt_df.pivot(index='prism_consumer_id', columns='category', values='cwt_monthly_habit').fillna(0)
        cwt_pivot_m.columns = [f"cat_{c}_cwt_monthly" for c in cwt_pivot_m.columns]
    else:
        cwt_pivot_w, cwt_pivot_m = pd.DataFrame(), pd.DataFrame()

    target_map = df.groupby('prism_consumer_id')['DQ_TARGET'].first()
    return weekly_pivot.join(monthly_pivot, how='outer').join(cwt_pivot_w, how='outer').join(cwt_pivot_m, how='outer').join(target_map, how='outer')

def build_global_tx_features(df_tx):
    df = df_tx.copy()
    df['abs_amount'] = df['amount'].abs()
    cat_counts = df.groupby(['prism_consumer_id', 'category']).size()
    total_counts = df.groupby('prism_consumer_id').size()
    count_props = cat_counts.div(total_counts, level='prism_consumer_id').unstack(fill_value=0)
    count_props.columns = [f"prop_count_cat_{c}" for c in count_props.columns]
    return count_props

def build_account_features(acctDF):
    agg_bal = acctDF.groupby("prism_consumer_id", as_index=False).agg(avg_balance=("balance", "mean"), max_balance=("balance", "max"), min_balance=("balance", "min"), num_accounts=("balance", "count"))
    pivot_bal = acctDF.pivot_table(index="prism_consumer_id", columns="account_type", values="balance", aggfunc="sum", fill_value=0).reset_index()
    pivot_bal["checking_to_savings_ratio"] = pivot_bal.get("CHECKING", 0) / (pivot_bal.get("SAVINGS", 0) + 1)
    return agg_bal, pivot_bal, pd.DataFrame()

def build_income_features(history_train):
    INCOME_CATS = ["DEPOSIT", "PAYCHECK", "INVESTMENT_INCOME"]
    income_txn = history_train[(history_train["credit_or_debit"] == "CREDIT") & (history_train["category"].isin(INCOME_CATS))].copy()
    avg_inc = income_txn.groupby("prism_consumer_id")["amount"].mean().reset_index(name="avg_monthly_income")
    return (avg_inc,)

def build_balance_dynamics(history_train):
    df = history_train.sort_values(["prism_consumer_id", "posted_date"]).copy()
    df["daily_change"] = df.groupby("prism_consumer_id")["running_balance"].diff()
    dyn = df.groupby("prism_consumer_id", as_index=False).agg(mean_daily_change=("daily_change", "mean"), std_daily_change=("daily_change", "std"))
    return (dyn,)

def build_balance_magnitude(history_train):
    return history_train.groupby('prism_consumer_id')['running_balance'].quantile([0.25, 0.5, 0.75]).unstack().reset_index()

def build_quality_of_drawdown(history_subset):
    df = history_subset.copy()
    cutoff_dates = df.groupby('prism_consumer_id')['posted_date'].transform('max') - pd.Timedelta(days=30)
    recent_30d = df[df['posted_date'] >= cutoff_dates]
    
    total_outflows = recent_30d[recent_30d['credit_or_debit'] == 'DEBIT'].groupby('prism_consumer_id')['amount'].sum()
    wealth_cats = ['SELF_TRANSFER', 'EXTERNAL_TRANSFER', 'INVESTMENT']
    wealth_outflows = recent_30d[(recent_30d['credit_or_debit'] == 'DEBIT') & 
                                 (recent_30d['category'].isin(wealth_cats))].groupby('prism_consumer_id')['amount'].sum()
    stress_cats = ['OVERDRAFT', 'ACCOUNT_FEES', 'SMALL_DOLLAR_ADVANCE']
    stress_outflows = recent_30d[(recent_30d['credit_or_debit'] == 'DEBIT') & 
                                 (recent_30d['category'].isin(stress_cats))].groupby('prism_consumer_id')['amount'].sum()
    drawdown_df = pd.DataFrame({
        'wealth_transfer_ratio': wealth_outflows / (total_outflows + 1e-6),
        'stress_outflow_ratio': stress_outflows / (total_outflows + 1e-6)
    }).reset_index()
    return drawdown_df

def build_cash_utilization(history_subset):
    df = history_subset.copy()
    total_debits = df[df['credit_or_debit'] == 'DEBIT'].groupby('prism_consumer_id')['amount'].sum()
    cash_debits = df[(df['credit_or_debit'] == 'DEBIT') & 
                     (df['category'] == 'ATM_CASH')].groupby('prism_consumer_id')['amount'].sum()
    cash_df = pd.DataFrame({
        'cash_utilization_pct': cash_debits / (total_debits + 1e-6)
    }).reset_index()
    return cash_df

def build_BNPL_utilization(history_subset):
    df = history_subset.copy()
    total_debits = df[df['credit_or_debit'] == 'DEBIT'].groupby('prism_consumer_id')['amount'].sum()
    BNPL_debits = df[(df['credit_or_debit'] == 'DEBIT') & 
                     (df['category'] == 'BNPL')].groupby('prism_consumer_id')['amount'].sum()
    BNPL_df = pd.DataFrame({
        'BNPL_utilization_pct': BNPL_debits / (total_debits + 1e-6)
    }).reset_index()
    return BNPL_df

def add_knn_error_features(X_train, y_train, X_test, train_probs):
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    hard_negatives = X_train[(y_train == 0) & (train_probs > 0.5)].copy()
    
    if hard_negatives.empty:
        X_train['dist_to_hard_neg'] = 0
        X_test['dist_to_hard_neg'] = 0
        return X_train, X_test

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    scaler = StandardScaler()
    hn_scaled = scaler.fit_transform(hard_negatives.fillna(0))
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)

    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(hn_scaled)

    dist_train, _ = knn.kneighbors(train_scaled)
    dist_test, _ = knn.kneighbors(test_scaled)

    X_train['dist_to_hard_neg'] = dist_train.mean(axis=1)
    X_test['dist_to_hard_neg'] = dist_test.mean(axis=1)
    
    return X_train, X_test


def extract_ALL_features(history_subset, acct_df):
    print("Extracting full habit matrix...")
    base_features = extract_consumer_habits(history_subset)
    agg_bal, pivot_bal, _ = build_account_features(acct_df) 
    income_feats = build_income_features(history_subset)
    balance_feats = build_balance_dynamics(history_subset)
    magnitude_feats = build_balance_magnitude(history_subset)
    drawdown = build_quality_of_drawdown(history_subset)
    cash = build_cash_utilization(history_subset)
    bnpl = build_BNPL_utilization(history_subset)
    global_tx_feats = build_global_tx_features(history_subset)
    
    features_df = base_features.copy()
    all_new_dfs = (agg_bal, pivot_bal, *income_feats, *balance_feats, magnitude_feats, drawdown, cash, bnpl, global_tx_feats) 
    
    for df in all_new_dfs:
        if 'prism_consumer_id' in df.columns:
            df = df.set_index('prism_consumer_id')
        features_df = features_df.join(df, how='left')
    features_df.columns = features_df.columns.astype(str)
    
    novel_functions = []
    for idx, func in enumerate(novel_functions, 1):
        try:
            novel_feats = func(history_subset, acct_df)
            if 'prism_consumer_id' in novel_feats.columns:
                novel_feats = novel_feats.set_index('prism_consumer_id')
            features_df = features_df.join(novel_feats, how='left')
        except Exception as e:
            print(f"Novel feature {idx} failed: {e}")
            
    # --- AUTO INJECTED ---
    try:
        novel_feats = build_novel_feature_1(history_subset, acct_df)
        if 'prism_consumer_id' in novel_feats.columns:
            novel_feats = novel_feats.set_index('prism_consumer_id')
        features_df = features_df.join(novel_feats, how='left')
    except Exception as e:
        print(f"Novel feature 1 failed: {e}")
    # ---------------------
    # --- AUTO INJECTED ---
    try:
        novel_feats = build_novel_feature_1(history_subset, acct_df)
        if 'prism_consumer_id' in novel_feats.columns:
            novel_feats = novel_feats.set_index('prism_consumer_id')
        features_df = features_df.join(novel_feats, how='left')
    except Exception as e:
        print(f"Novel feature 1 failed: {e}")
    # ---------------------
    return features_df.reset_index()

def build_novel_feature_1(history_subset, acct_df):
    df = history_subset.copy()
    
    # Calculate average transaction amount per consumer
    avg_amount = df.groupby('prism_consumer_id')['amount'].mean().reset_index(name='avg_transaction_amount')
    
    # Merge with account balance data
    merged_df = avg_amount.merge(acct_df[['prism_consumer_id', 'balance']], on='prism_consumer_id', how='left')
    
    # Calculate transaction to balance ratio
    merged_df['transaction_to_balance_ratio'] = (merged_df['avg_transaction_amount'] / (merged_df['balance'] + 1e-6)).fillna(0)
    
    # Return the new feature as a DataFrame
    return merged_df[['prism_consumer_id', 'transaction_to_balance_ratio']]


def build_novel_feature_1(history_subset, acct_df):
    df = history_subset.copy()
    
    # Calculate the total number of transactions per consumer
    txn_count = df.groupby('prism_consumer_id').size().reset_index(name='total_transactions')
    
    # Merge with account balance data
    merged_df = txn_count.merge(acct_df[['prism_consumer_id', 'balance']], on='prism_consumer_id', how='left')
    
    # Calculate transactions per balance ratio
    merged_df['transactions_per_balance_ratio'] = (merged_df['total_transactions'] / (merged_df['balance'] + 1e-6)).fillna(0)
    
    # Return the new feature as a DataFrame
    return merged_df[['prism_consumer_id', 'transactions_per_balance_ratio']]
