import pandas as pd

def build_novel_feature(history_subset, acct_df):
    # Calculate average transaction frequency per consumer
    df = history_subset.copy()
    df['transaction_date'] = pd.to_datetime(df['posted_date']).dt.date
    freq = df.groupby(['prism_consumer_id', 'transaction_date']).size().reset_index(name='freq')
    avg_freq = freq.groupby('prism_consumer_id')['freq'].mean().reset_index(name='avg_transaction_frequency')
    
    # Merge with account data to get account count per consumer
    agg_bal, _, _ = build_account_features(acct_df)
    avg_freq = avg_freq.merge(agg_bal[['prism_consumer_id', 'num_accounts']], on='prism_consumer_id', how='left')

    # Calculate the novel feature: average transaction frequency per account
    avg_freq['avg_tx_per_account'] = avg_freq['avg_transaction_frequency'] / (avg_freq['num_accounts'] + 1e-6)
    
    return avg_freq[['prism_consumer_id', 'avg_tx_per_account']]