import pandas as pd
from load_data import load_data
from features.haris.income_feats import income_feats
from features.haris.balance_feats import balance_feats
from features.haris.category_feats import category_feats

def run_pipeline():
    # Load Data
    print("Loading data...")
    consDF, testDF, acctDF, trxnDF, cat_map = load_data()

    # Income features
    print("Generating income features...")
    features = income_feats(trxnDF)

    # Balance Features
    print("Generating balance features...")
    features = features.merge(balance_feats(acctDF), on='prism_consumer_id')

    # Income + Balance Features
    features['balance_income_ratio'] = features['avg_balance'] / (features['avg_monthly_income'])

    # Category Features
    # print("Generating category features...")
    # features = features.merge(category_feats(trxnDF), on='prism_consumer_id')
        
    features[features.isna()] = 0

    df = consDF.merge(features, on='prism_consumer_id')

    return df

if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())