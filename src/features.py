import pandas as pd
import numpy as np
import pywt

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

def build_spending_volatility_and_stress(history_subset):
    df = history_subset.copy()
    df['daily_abs_change'] = df.groupby('prism_consumer_id')['running_balance'].diff().abs()
    vola = df.groupby('prism_consumer_id')['daily_abs_change'].std().reset_index(name='spending_volatility')
    overdrafts = df[df['category'] == 'OVERDRAFT'].groupby('prism_consumer_id').size().reset_index(name='overdraft_count')
    total_debits = df[df['credit_or_debit'] == 'DEBIT'].groupby('prism_consumer_id').size().reset_index(name='total_debits')
    stress = pd.merge(overdrafts, total_debits, on='prism_consumer_id', how='right').fillna(0)
    stress['liquidity_stress_ratio'] = stress['overdraft_count'] / (stress['total_debits'] + 1e-6)
    final_feats = pd.merge(vola, stress[['prism_consumer_id', 'liquidity_stress_ratio']], on='prism_consumer_id', how='outer')
    return final_feats

def build_monthly_velocity(history_subset):
    df = history_subset.copy()
    df['transaction_month'] = df['posted_date'].dt.to_period('M')
    txn_freq = df.groupby(['prism_consumer_id', 'transaction_month']).size().reset_index(name='monthly_txn_count')
    avg_txn_freq = txn_freq.groupby('prism_consumer_id')['monthly_txn_count'].mean().reset_index(name='avg_txn_freq_per_month')
    return avg_txn_freq

def build_transaction_ratios(history_subset, acct_df):
    df = history_subset.copy()
    txn_count = df.groupby('prism_consumer_id').size().reset_index(name='total_transactions')
    lifespan = df.groupby('prism_consumer_id')['posted_date'].agg(lambda x: (x.max() - x.min()).days + 1).reset_index(name='days_observed')
    merged = txn_count.merge(lifespan, on='prism_consumer_id')
    merged['txns_per_day'] = merged['total_transactions'] / merged['days_observed']
    merged = merged.merge(acct_df[['prism_consumer_id', 'balance']], on='prism_consumer_id', how='left')
    merged['txn_rate_to_balance_ratio'] = (merged['txns_per_day'] / (merged['balance'] + 1e-6)).fillna(0)
    return merged[['prism_consumer_id', 'txn_rate_to_balance_ratio']]


# ==========================================
# --- PREVIOUS AI DISCOVERED FEATURES ---
# ==========================================

def build_nonlinear_ratio(history_subset, acct_df):
    df = history_subset.copy()
    df['balance_to_amount_ratio'] = (np.abs(df['running_balance']) / (np.abs(df['amount']) + 1e-6)).fillna(0)
    df['nonlinear_transformed_ratio'] = np.cos(np.tanh(df['balance_to_amount_ratio'] * 2) * np.pi).fillna(0)
    result_df = df.groupby('prism_consumer_id')['nonlinear_transformed_ratio'].mean().reset_index()
    return result_df[['prism_consumer_id', 'nonlinear_transformed_ratio']]

def build_stress_intensity(history_subset, acct_df):
    if history_subset.empty or acct_df.empty:
        return pd.DataFrame(columns=['prism_consumer_id', 'stress_intensity_asset_ratio'])
        
    df = history_subset.copy()
    a_df = acct_df.copy()
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)
    
    stress_mask = df['running_balance'] <= 0
    stress_agg = df[stress_mask].groupby('prism_consumer_id')['signed_amount'].agg(mean_stress_spending='mean').reset_index()
    stress_agg['abs_stress'] = np.abs(stress_agg['mean_stress_spending'])
    
    asset_agg = a_df.groupby('prism_consumer_id')['balance'].agg(avg_asset_buffer='mean').reset_index()
    unique_consumers = df['prism_consumer_id'].unique()
    
    result = pd.merge(pd.DataFrame({'prism_consumer_id': unique_consumers}), stress_agg[['prism_consumer_id', 'abs_stress']], on='prism_consumer_id', how='left')
    result['abs_stress'] = result['abs_stress'].fillna(0.0)
    
    merged_df = pd.merge(result, asset_agg, on='prism_consumer_id', how='left')
    merged_df['avg_asset_buffer'] = merged_df['avg_asset_buffer'].fillna(1.0)
    merged_df['raw_intensity_ratio'] = merged_df['abs_stress'] / (merged_df['avg_asset_buffer'] + 1e-6)
    merged_df['stress_intensity_asset_ratio'] = np.tanh(merged_df['raw_intensity_ratio']).astype('float32')
    merged_df['stress_intensity_asset_ratio'] = merged_df['stress_intensity_asset_ratio'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    if merged_df['prism_consumer_id'].duplicated().any():
        merged_df = merged_df.drop_duplicates(subset=['prism_consumer_id'], keep='first')
    return merged_df[['prism_consumer_id', 'stress_intensity_asset_ratio']]

def build_state_switching_volatility(history_subset, acct_df):
    if history_subset.empty:
        return pd.DataFrame({'prism_consumer_id': [], 'novel_balance_state_switching_volatility': []})
        
    df = history_subset.copy()
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    
    df['state_negative'] = (df['running_balance'] <= 0).astype(int)
    df['prev_state'] = df.groupby('prism_consumer_id')['state_negative'].shift(1)
    df['is_transition'] = (df['state_negative'] != df['prev_state']).astype(int)
    
    transition_stats = df.groupby('prism_consumer_id').agg(num_transitions=('is_transition', 'sum'), total_transactions=('state_negative', 'count')).reset_index()
    transition_stats['switching_rate'] = transition_stats['num_transitions'] / (transition_stats['total_transactions'] + 1e-6)
    
    if not acct_df.empty:
        a_df = acct_df.copy()
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)
        acct_agg = a_df.groupby('prism_consumer_id')['balance'].mean().reset_index()
        acct_agg.columns = ['prism_consumer_id', 'avg_account_balance']
        merged_stats = pd.merge(transition_stats, acct_agg, on='prism_consumer_id', how='left')
        merged_stats['avg_account_balance'] = merged_stats['avg_account_balance'].fillna(1.0)
    else:
        merged_stats = transition_stats.copy()
        merged_stats['avg_account_balance'] = 1.0
        
    merged_stats['wealth_normalized_switching'] = (merged_stats['switching_rate'] * 10.0) / (np.abs(merged_stats['avg_account_balance']) + 1e-6)
    merged_stats['novel_balance_state_switching_volatility'] = np.tanh(merged_stats['wealth_normalized_switching']).astype('float32')
    
    result_df = merged_stats[['prism_consumer_id', 'novel_balance_state_switching_volatility']].copy()
    result_df['novel_balance_state_switching_volatility'] = result_df['novel_balance_state_switching_volatility'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    if result_df['prism_consumer_id'].duplicated().any():
        result_df = result_df.drop_duplicates(subset=['prism_consumer_id'], keep='first')
    return result_df.reset_index(drop=True)

def build_volatility_regime_shift(history_subset, acct_df):
    if history_subset.empty:
        return pd.DataFrame(columns=['prism_consumer_id', 'recent_volatility_regime_shift_index'])
        
    df = history_subset.copy()
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    df['abs_balance_change'] = df.groupby('prism_consumer_id')['running_balance'].diff().fillna(0).abs()
    
    max_dates = df.groupby('prism_consumer_id')['posted_date'].max().reset_index()
    df = df.merge(max_dates, on='prism_consumer_id', how='left', suffixes=('', '_max'))
    cutoff_date = df['posted_date_max'] - pd.Timedelta(days=90)
    df['is_recent'] = df['posted_date'] >= cutoff_date
    
    recent_vol = df[df['is_recent']].groupby('prism_consumer_id')['abs_balance_change'].mean().reset_index(name='recent_mean_vol')
    hist_vol = df[~df['is_recent']].groupby('prism_consumer_id')['abs_balance_change'].mean().reset_index(name='historical_mean_vol')
    merged_stats = pd.merge(recent_vol, hist_vol, on='prism_consumer_id', how='outer').fillna(0)
    merged_stats['volatility_ratio_raw'] = merged_stats['recent_mean_vol'] / (merged_stats['historical_mean_vol'] + 1e-6)
    
    if not acct_df.empty:
        a_df = acct_df.copy()
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)
        acct_agg = a_df.groupby('prism_consumer_id')['balance'].mean().reset_index(name='avg_account_balance')
        merged_stats = pd.merge(merged_stats, acct_agg, on='prism_consumer_id', how='left')
        merged_stats['avg_account_balance'] = merged_stats['avg_account_balance'].fillna(1.0)
    else:
        merged_stats['avg_account_balance'] = 1.0
        
    merged_stats['wealth_adjusted_score'] = merged_stats['volatility_ratio_raw'] / (np.abs(merged_stats['avg_account_balance']) + 1e-6)
    merged_stats['recent_volatility_regime_shift_index'] = np.tanh(merged_stats['wealth_adjusted_score']).astype('float32')
    
    result_df = merged_stats[['prism_consumer_id', 'recent_volatility_regime_shift_index']].copy()
    result_df['recent_volatility_regime_shift_index'] = result_df['recent_volatility_regime_shift_index'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    if result_df['prism_consumer_id'].duplicated().any():
        result_df = result_df.drop_duplicates(subset=['prism_consumer_id'], keep='first')
    return result_df.reset_index(drop=True)


# ==========================================
# --- NEW ADVANCED AGENT DISCOVERIES ---
# ==========================================

def build_recovery_latency_normalized(history_subset, acct_df):
    if history_subset.empty:
        return pd.DataFrame(columns=['prism_consumer_id', 'recovery_latency_normalized'])
    
    df = history_subset.copy()
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    
    # Identify recovery events: negative balance transitions to positive within n days
    df['is_negative'] = (df['running_balance'] < 0).astype(int)
    df['prev_negative'] = df.groupby('prism_consumer_id')['is_negative'].shift(1)
    
    # Detect entry into negative territory
    df['entered_negative'] = ((df['is_negative'] == 1) & (df['prev_negative'] == 0)).astype(int)
    
    # Track recovery: time spent negative until balance turns positive
    recovery_lags = []
    
    for consumer_id, group in df.groupby('prism_consumer_id'):
        entered_indices = group[group['entered_negative'] == 1].index.tolist()
        
        for entry_idx in entered_indices:
            sub_group = group.loc[entry_idx:].copy()
            recovery_found = False
            
            for i, row in sub_group.iterrows():
                if row['is_negative'] == 0 and row['running_balance'] > 0:
                    lag_days = (row['posted_date'] - group.loc[entry_idx]['posted_date']).days
                    recovery_lags.append({'prism_consumer_id': consumer_id, 'recovery_lag_days': max(1, min(lag_days, 365))})
                    recovery_found = True
                    break
            
            if not recovery_found:
                # If no recovery within window, cap at maximum observed lag
                last_row = group.loc[entry_idx:].iloc[-1]
                lag_days = max(1, (last_row['posted_date'] - group.loc[entry_idx]['posted_date']).days)
                recovery_lags.append({'prism_consumer_id': consumer_id, 'recovery_lag_days': min(lag_days, 365)})
    
    if not recovery_lags:
        return pd.DataFrame(columns=['prism_consumer_id', 'recovery_latency_normalized'])
    
    recovery_df = pd.DataFrame(recovery_lags)
    avg_recovery_lag = recovery_df.groupby('prism_consumer_id')['recovery_lag_days'].mean().reset_index(name='avg_recovery_lag_days')
    
    # Account size normalization for cross-consumer comparability
    if not acct_df.empty:
        a_df = acct_df.copy()
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)
        asset_avg = a_df.groupby('prism_consumer_id')['balance'].mean().reset_index(name='asset_buffer')
        merged = pd.merge(avg_recovery_lag, asset_avg, on='prism_consumer_id', how='left')
        merged['asset_buffer'] = merged['asset_buffer'].fillna(1.0).clip(lower=1.0)
    else:
        merged = avg_recovery_lag.copy()
        merged['asset_buffer'] = 1.0
    
    # Normalize recovery time by asset buffer to create dimensionless index
    merged['recovery_latency_raw'] = merged['avg_recovery_lag_days'] / (merged['asset_buffer'] + 1e-6)
    merged['recovery_latency_normalized'] = np.tanh(merged['recovery_latency_raw']).astype('float32')
    merged['recovery_latency_normalized'] = merged['recovery_latency_normalized'].clip(-1.0, 1.0).fillna(0.0)
    
    # Remove potential infinity values
    merged['recovery_latency_normalized'] = merged['recovery_latency_normalized'].replace([np.inf, -np.inf], 0.0)
    
    result_df = merged[['prism_consumer_id', 'recovery_latency_normalized']].drop_duplicates(subset=['prism_consumer_id'], keep='first')
    return result_df.reset_index(drop=True)


def build_balance_autocorr_lag1_index(history_subset, acct_df):
    result_cols = ['prism_consumer_id', 'balance_autocorr_lag1_index']
    if history_subset.empty:
        return pd.DataFrame(columns=result_cols)

    df = history_subset.copy()
    a_df = acct_df.copy() if not acct_df.empty else pd.DataFrame()
    
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    if not a_df.empty:
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)

    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    df['lagged_balance'] = df.groupby('prism_consumer_id')['running_balance'].shift(1)
    df_valid = df.dropna(subset=['lagged_balance']).copy()
    
    if df_valid.empty:
        return pd.DataFrame({'prism_consumer_id': history_subset['prism_consumer_id'].astype(str).unique(), 
                             'balance_autocorr_lag1_index': [0.0] * len(history_subset['prism_consumer_id'].astype(str).unique())})

    df_valid['x_val'] = df_valid['running_balance']
    df_valid['y_val'] = df_valid['lagged_balance']
    df_valid['x_sq'] = df_valid['x_val'] ** 2
    df_valid['y_sq'] = df_valid['y_val'] ** 2
    df_valid['xy_prod'] = df_valid['x_val'] * df_valid['y_val']
    
    corr_stats = df_valid.groupby('prism_consumer_id').agg(
        n=('x_val', 'count'),
        sum_x=('x_val', 'sum'),
        sum_y=('y_val', 'sum'),
        sum_xy=('xy_prod', 'sum'),
        sum_x2=('x_sq', 'sum'),
        sum_y2=('y_sq', 'sum')
    ).reset_index()

    numerator = corr_stats['n'] * corr_stats['sum_xy'] - (corr_stats['sum_x'] * corr_stats['sum_y'])
    var_x = np.maximum((corr_stats['n'] * corr_stats['sum_x2']) - (corr_stats['sum_x'] ** 2), 0.0)
    var_y = np.maximum((corr_stats['n'] * corr_stats['sum_y2']) - (corr_stats['sum_y'] ** 2), 0.0)
    
    denom = np.sqrt(var_x * var_y + 1e-6)
    corr_raw = numerator / (denom + 1e-6)
    corr_raw = np.clip(corr_raw, -1.0, 1.0)
    corr_stats['corr_raw'] = corr_raw
    
    if not a_df.empty:
        a_df['abs_balance'] = a_df['balance'].abs()
        asset_agg = a_df.groupby('prism_consumer_id').agg(mean_abs_balance=('abs_balance', 'mean')).reset_index()
        merged_stats = pd.merge(corr_stats, asset_agg, on='prism_consumer_id', how='left')
    else:
        merged_stats = corr_stats
        merged_stats['mean_abs_balance'] = 100.0
        
    merged_stats['mean_abs_balance'] = merged_stats['mean_abs_balance'].fillna(1e-6)
    merged_stats['wealth_adjusted_score'] = merged_stats['corr_raw'] / (merged_stats['mean_abs_balance'] + 1e-6)
    merged_stats['balance_autocorr_lag1_index'] = np.tanh(merged_stats['wealth_adjusted_score']).astype('float32')
    merged_stats['balance_autocorr_lag1_index'] = merged_stats['balance_autocorr_lag1_index'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    final_df = merged_stats[['prism_consumer_id', 'balance_autocorr_lag1_index']].drop_duplicates(subset=['prism_consumer_id'], keep='first')
    final_df['prism_consumer_id'] = final_df['prism_consumer_id'].astype(str)
    return final_df.reset_index(drop=True)


def build_balance_dist_skewness_index(history_subset, acct_df):
    result_cols = ['prism_consumer_id', 'balance_dist_skewness_index']
    if history_subset.empty:
        return pd.DataFrame(columns=result_cols)
        
    df = history_subset.copy()
    a_df = acct_df.copy() if not acct_df.empty else pd.DataFrame()
    
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    if not a_df.empty:
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)

    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    
    def compute_skewness(group):
        bal = group['running_balance'].values
        n = len(bal)
        if n < 3: return np.nan
        mean_val = np.mean(bal)
        std_val = np.std(bal, ddof=1)
        if std_val < 1e-8: return 0.0
        sum_diff_cubed = np.sum(np.power(bal - mean_val, 3))
        skewness = (sum_diff_cubed / n) / (std_val ** 3)
        return float(skewness)

    skew_df = df.groupby('prism_consumer_id').apply(compute_skewness).reset_index()
    skew_df.columns = ['prism_consumer_id', 'raw_skewness']
    valid_skew = skew_df.dropna(subset=['raw_skewness'])
    
    if len(valid_skew) == 0:
        unique_ids = df['prism_consumer_id'].astype(str).unique()
        return pd.DataFrame({'prism_consumer_id': unique_ids, 'balance_dist_skewness_index': [0.0] * len(unique_ids)})

    if not a_df.empty:
        a_df['abs_balance'] = np.abs(a_df['balance']) + 1e-6
        asset_agg = a_df.groupby('prism_consumer_id').agg(mean_abs_asset=('abs_balance', 'mean')).reset_index()
        merged_stats = pd.merge(valid_skew, asset_agg, on='prism_consumer_id', how='left')
    else:
        merged_stats = valid_skew.copy()
        merged_stats['mean_abs_asset'] = 10.0
        
    merged_stats['mean_abs_asset'] = merged_stats['mean_abs_asset'].fillna(1e-6).clip(lower=1e-6)
    merged_stats['wealth_adjusted_score'] = np.abs(merged_stats['raw_skewness']) / (merged_stats['mean_abs_asset'] + 1e-6)
    merged_stats['balance_dist_skewness_index'] = np.tanh(merged_stats['wealth_adjusted_score']).astype('float32')
    merged_stats['balance_dist_skewness_index'] = merged_stats['balance_dist_skewness_index'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    final_df = merged_stats[['prism_consumer_id', 'balance_dist_skewness_index']].drop_duplicates(subset=['prism_consumer_id'], keep='first')
    final_df['prism_consumer_id'] = final_df['prism_consumer_id'].astype(str)
    return final_df.reset_index(drop=True)


def build_balance_path_efficiency_index(history_subset, acct_df):
    result_cols = ['prism_consumer_id', 'balance_path_efficiency_index']
    if history_subset.empty:
        return pd.DataFrame(columns=result_cols)

    df = history_subset.copy()
    a_df = acct_df.copy() if not acct_df.empty else None
    
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    
    if a_df is not None:
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)

    df['daily_change'] = df.groupby('prism_consumer_id')['running_balance'].diff()
    df['abs_daily_change'] = np.abs(df['daily_change'])

    tv_stats = df.groupby('prism_consumer_id')['abs_daily_change'].sum().reset_index(name='total_variation')
    first_bal = df.groupby('prism_consumer_id')['running_balance'].first().reset_index(name='start_balance')
    last_bal = df.groupby('prism_consumer_id')['running_balance'].last().reset_index(name='end_balance')
    
    net_displacement = pd.merge(first_bal, last_bal, on='prism_consumer_id', how='inner')
    net_displacement['net_disp'] = np.abs(net_displacement['start_balance'] - net_displacement['end_balance'])
    net_stats = net_displacement[['prism_consumer_id', 'net_disp']]
    
    merged_stats = pd.merge(tv_stats, net_stats, on='prism_consumer_id', how='left')
    
    if merged_stats.empty:
        unique_ids = df['prism_consumer_id'].astype(str).unique()
        return pd.DataFrame({'prism_consumer_id': unique_ids, 'balance_path_efficiency_index': [0.0] * len(unique_ids)})
    
    merged_stats['efficiency_ratio_raw'] = merged_stats['total_variation'] / (merged_stats['net_disp'] + 1e-6)
    
    if a_df is not None and not a_df.empty:
        a_df['abs_balance'] = np.abs(a_df['balance']) + 1e-6
        asset_agg = a_df.groupby('prism_consumer_id')['abs_balance'].mean().reset_index(name='mean_abs_asset')
        merged_stats = pd.merge(merged_stats, asset_agg, on='prism_consumer_id', how='left')
    else:
        merged_stats['mean_abs_asset'] = 100.0
        
    merged_stats['mean_abs_asset'] = merged_stats['mean_abs_asset'].fillna(1e-6).clip(lower=1e-6)
    merged_stats['wealth_adjusted_score'] = merged_stats['efficiency_ratio_raw'] / (merged_stats['mean_abs_asset'] + 1e-6)
    merged_stats['balance_path_efficiency_index'] = np.tanh(merged_stats['wealth_adjusted_score']).astype('float32')
    merged_stats['balance_path_efficiency_index'] = merged_stats['balance_path_efficiency_index'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    final_df = merged_stats[['prism_consumer_id', 'balance_path_efficiency_index']].drop_duplicates(subset=['prism_consumer_id'], keep='first')
    final_df['prism_consumer_id'] = final_df['prism_consumer_id'].astype(str)
    return final_df.reset_index(drop=True)


def build_asymmetry_recovery_flow_index(history_subset, acct_df):
    result_cols = ['prism_consumer_id', 'asymmetry_recovery_flow_index']
    if history_subset.empty:
        return pd.DataFrame(columns=result_cols)
        
    df = history_subset.copy()
    a_df = acct_df.copy() if not acct_df.empty else None
    
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    if a_df is not None:
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)

    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    
    try:
        median_bal = df.groupby('prism_consumer_id')['running_balance'].transform('median')
    except Exception:
        median_bal = pd.Series(dtype=float)
    
    if median_bal.empty or len(median_bal) == 0:
        unique_ids = df['prism_consumer_id'].astype(str).unique()
        return pd.DataFrame({'prism_consumer_id': unique_ids, 'asymmetry_recovery_flow_index': [0.0] * len(unique_ids)})

    df['median_running'] = median_bal
    low_regime_mask = df['running_balance'] < df['median_running']
    df['daily_flow'] = df.groupby('prism_consumer_id')['running_balance'].diff().fillna(0.0)
    low_regime_df = df[low_regime_mask].copy()
    
    if len(low_regime_df) == 0:
        unique_ids = df['prism_consumer_id'].astype(str).unique()
        return pd.DataFrame({'prism_consumer_id': unique_ids, 'asymmetry_recovery_flow_index': [0.0] * len(unique_ids)})

    flow_stats = low_regime_df.groupby('prism_consumer_id').agg(
        total_upflow=('daily_flow', lambda x: max(x[x > 0].sum(), 0.0)),
        total_downflow=('daily_flow', lambda x: abs(min(x[x < 0].sum(), 0.0)))
    ).reset_index()
    
    flow_stats['recovery_ratio_raw'] = flow_stats['total_upflow'] / (flow_stats['total_downflow'] + 1e-6)
    
    if a_df is not None and not a_df.empty:
        a_df['abs_balance'] = np.abs(a_df['balance']) + 1e-6
        asset_agg = a_df.groupby('prism_consumer_id').agg(mean_abs_asset=('abs_balance', 'mean')).reset_index()
        merged_stats = pd.merge(flow_stats, asset_agg, on='prism_consumer_id', how='left')
    else:
        merged_stats = flow_stats.copy()
        merged_stats['mean_abs_asset'] = 100.0
        
    merged_stats['mean_abs_asset'] = merged_stats['mean_abs_asset'].fillna(1e-6).clip(lower=1e-6)
    merged_stats['wealth_adjusted_score'] = merged_stats['recovery_ratio_raw'] / (merged_stats['mean_abs_asset'] + 1e-6)
    merged_stats['asymmetry_recovery_flow_index'] = np.tanh(merged_stats['wealth_adjusted_score']).astype('float32')
    merged_stats['asymmetry_recovery_flow_index'] = merged_stats['asymmetry_recovery_flow_index'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    final_df = merged_stats[['prism_consumer_id', 'asymmetry_recovery_flow_index']].drop_duplicates(subset=['prism_consumer_id'], keep='first')
    final_df['prism_consumer_id'] = final_df['prism_consumer_id'].astype(str)
    return final_df.reset_index(drop=True)


def build_volatility_concentration_ratio(history_subset, acct_df):
    result_cols = ['prism_consumer_id', 'volatility_concentration_ratio']
    if history_subset.empty:
        return pd.DataFrame(columns=result_cols)

    df = history_subset.copy()
    a_df = acct_df.copy() if not acct_df.empty else None
    
    df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
    if a_df is not None:
        a_df['prism_consumer_id'] = a_df['prism_consumer_id'].astype(str)

    df = df.sort_values(['prism_consumer_id', 'posted_date']).reset_index(drop=True)
    df['daily_abs_change'] = df.groupby('prism_consumer_id')['running_balance'].diff().abs()
    
    df_valid = df.dropna(subset=['daily_abs_change']).copy()
    
    if df_valid.empty:
        unique_ids = df['prism_consumer_id'].astype(str).unique()
        return pd.DataFrame({'prism_consumer_id': unique_ids, 'volatility_concentration_ratio': [0.0] * len(unique_ids)})

    median_daily = df_valid.groupby('prism_consumer_id')['daily_abs_change'].transform('median')
    df_valid['is_high_impact'] = df_valid['daily_abs_change'] > median_daily
    
    total_var_stats = df_valid.groupby('prism_consumer_id')['daily_abs_change'].sum().reset_index(name='total_variation_sum')
    high_impact_mask = df_valid['is_high_impact'] == True
    high_impact_stats = df_valid[high_impact_mask].groupby('prism_consumer_id')['daily_abs_change'].sum().reset_index(name='high_impact_sum')
    
    merged_stats = pd.merge(total_var_stats, high_impact_stats, on='prism_consumer_id', how='left')
    
    if merged_stats.empty:
        unique_ids = df['prism_consumer_id'].astype(str).unique()
        return pd.DataFrame({'prism_consumer_id': unique_ids, 'volatility_concentration_ratio': [0.0] * len(unique_ids)})

    merged_stats['high_impact_sum'] = merged_stats['high_impact_sum'].fillna(0.0)
    merged_stats['conc_raw'] = merged_stats['high_impact_sum'] / (merged_stats['total_variation_sum'] + 1e-6)
    
    if a_df is not None and not a_df.empty:
        a_df['abs_balance'] = np.abs(a_df['balance']) + 1e-6
        asset_agg = a_df.groupby('prism_consumer_id').agg(mean_abs_asset=('abs_balance', 'mean')).reset_index()
        merged_final = pd.merge(merged_stats, asset_agg, on='prism_consumer_id', how='left')
    else:
        merged_final = merged_stats.copy()
        merged_final['mean_abs_asset'] = 10.0
        
    merged_final['mean_abs_asset'] = merged_final['mean_abs_asset'].fillna(1e-6).clip(lower=1e-6)
    merged_final['wealth_adjusted_score'] = merged_final['conc_raw'] / (merged_final['mean_abs_asset'] + 1e-6)
    merged_final['volatility_concentration_ratio'] = np.tanh(merged_final['wealth_adjusted_score']).astype('float32')
    merged_final['volatility_concentration_ratio'] = merged_final['volatility_concentration_ratio'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    final_df = merged_final[['prism_consumer_id', 'volatility_concentration_ratio']].drop_duplicates(subset=['prism_consumer_id'], keep='first')
    final_df['prism_consumer_id'] = final_df['prism_consumer_id'].astype(str)
    return final_df.reset_index(drop=True)


# ==========================================
# --- MASTER EXTRACTION PIPELINE ---
# ==========================================

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
    volatility_feats = build_spending_volatility_and_stress(history_subset)
    velocity_feats = build_monthly_velocity(history_subset)
    ratio_feats = build_transaction_ratios(history_subset, acct_df)
    
    # Previous Discoveries
    ai_nonlinear = build_nonlinear_ratio(history_subset, acct_df)
    ai_stress = build_stress_intensity(history_subset, acct_df)
    ai_state_switch = build_state_switching_volatility(history_subset, acct_df)
    ai_regime_shift = build_volatility_regime_shift(history_subset, acct_df)
    
    # NEW Agent Discoveries!
    ai_rec_latency = build_recovery_latency_normalized(history_subset, acct_df)
    ai_autocorr = build_balance_autocorr_lag1_index(history_subset, acct_df)
    ai_skewness = build_balance_dist_skewness_index(history_subset, acct_df)
    ai_path_eff = build_balance_path_efficiency_index(history_subset, acct_df)
    ai_asym_flow = build_asymmetry_recovery_flow_index(history_subset, acct_df)
    ai_vol_conc = build_volatility_concentration_ratio(history_subset, acct_df)
    
    features_df = base_features.copy()
    
    all_new_dfs = (
        agg_bal, pivot_bal, *income_feats, *balance_feats, 
        magnitude_feats, drawdown, cash, bnpl, global_tx_feats,
        volatility_feats, velocity_feats, ratio_feats,
        ai_nonlinear, ai_stress, ai_state_switch, ai_regime_shift,
        ai_rec_latency, ai_autocorr, ai_skewness, ai_path_eff, ai_asym_flow, ai_vol_conc
    ) 
    
    for df in all_new_dfs:
        if 'prism_consumer_id' in df.columns:
            # Ensure type alignment for smooth indexing and joins
            df['prism_consumer_id'] = df['prism_consumer_id'].astype(str)
            df = df.set_index('prism_consumer_id')
        features_df = features_df.join(df, how='left')
        
    features_df.columns = features_df.columns.astype(str)
    return features_df.reset_index()