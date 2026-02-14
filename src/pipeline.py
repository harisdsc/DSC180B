from src.load_data import load_data
from src.features.haris.income_feats import income_feats
from src.features.haris.balance_feats import balance_feats
from src.features.haris.category_feats import category_feats
from src.features.ada.income_ratio_features import income_ratio_feats
from src.features.brighton.account_feats import build_account_features
from src.features.brighton.income_feats import build_income_features
from src.features.brighton.running_balance import build_running_balance
from src.features.brighton.balance_feats import build_balance_dynamics
import pandas as pd
import os

def run_pipeline():
    if os.path.exists('data/features.pqt'):
        print("Loading precomputed features...")
        df = pd.read_parquet('data/features.pqt')
        print(f"Shape: {df.shape}")
        return df
    
    # Load Data
    print("Loading data...")
    consDF, acctDF, trxnDF, cat_map = load_data()

    features = consDF[['prism_consumer_id']]

    # Income features
    print("Generating income features...")
    income_df_haris = income_feats(trxnDF)
    features = features.merge(income_df_haris, on='prism_consumer_id', how='left')
    income_df_ada = income_ratio_feats(trxnDF, cat_map)
    features = features.merge(income_df_ada, on='prism_consumer_id', how='left')
    income_df_brighton = build_income_features(trxnDF, cat_map)
    features = features.merge(income_df_brighton, on='prism_consumer_id', how='left')

    # Balance Features
    print("Generating balance features...")
    balance_df_haris = balance_feats(acctDF)
    features = features.merge(balance_df_haris, on='prism_consumer_id', how='left')
    features['avg_balance'] = features['avg_balance'].fillna(0)
    full_balance_df = build_running_balance(acctDF, trxnDF)
    balance_df_brighton = build_balance_dynamics(full_balance_df)
    features = features.merge(balance_df_brighton, on='prism_consumer_id', how='left')

    # Income + Balance Features
    features['balance_income_ratio'] = features.apply(
        lambda x: x['avg_balance'] / x['avg_monthly_income'] if x['avg_monthly_income'] > 0 else 0, axis=1
    )

    # Account features
    print("Generating account features...")
    account_df_brighton = build_account_features(acctDF)
    features = features.merge(account_df_brighton, on='prism_consumer_id', how='left')

    # Category Features
    # print("Generating category features...")
    # cat_df = category_feats(trxnDF)
    # features = features.merge(cat_df, on='prism_consumer_id', how='left')

    df = consDF.merge(features, on='prism_consumer_id', how='inner')

    df = df.fillna(0)

    print(f"Shape: {df.shape}")
    print(df.head())

    df.to_parquet('data/features.pqt')
    return df

if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())
    print(f"Final shape: {df.shape}")
