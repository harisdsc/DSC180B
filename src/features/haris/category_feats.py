import pandas as pd

def category_feats(trxnDF):
    neg_cats = [25, 10, 46, 23]
    neg_trxns = trxnDF[trxnDF['category'].isin(neg_cats)].copy()
    monthly_neg_count = neg_trxns.groupby(['prism_consumer_id', neg_trxns['posted_date'].dt.to_period('M')])['amount'].count().reset_index()
    avg_monthly_neg_count = monthly_neg_count.groupby('prism_consumer_id')['amount'].mean().rename('avg_monthly_neg_count')
    total_neg_amount = neg_trxns.groupby('prism_consumer_id')['amount'].sum().rename('total_neg_amount')
    has_neg = neg_trxns.groupby('prism_consumer_id')['amount'].any().astype(int).rename('has_neg')
    return pd.concat([avg_monthly_neg_count, total_neg_amount, has_neg], axis=1).reset_index()




