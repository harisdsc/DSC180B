import pandas as pd

def income_feats(trxnDF): 
    income_cats = [2, 3, 7, 8, 9, 49]
    income_trxns = trxnDF[trxnDF['category'].isin(income_cats)].copy()
    income_trxns['posted_date'] = pd.to_datetime(income_trxns['posted_date'])
    monthly_income = income_trxns.groupby(
        ['prism_consumer_id', income_trxns['posted_date'].dt.to_period('M')])['amount'].sum()
    avg_monthly_income = monthly_income.groupby('prism_consumer_id').mean().rename('avg_monthly_income')
    income_std = monthly_income.groupby('prism_consumer_id').std().rename('income_std')
    features = pd.concat([avg_monthly_income, income_std], axis=1).reset_index()
    return features
    