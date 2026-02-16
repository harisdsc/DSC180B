import pandas as pd

def balance_feats(acctDF):
    avg_balance = acctDF.groupby('prism_consumer_id')['balance'].mean().rename('avg_balance')
    features = avg_balance.reset_index()
    features = features.fillna(0)
    return features