from src.load_data import load_data
from src.features.haris.income_feats import income_feats
from src.features.haris.balance_feats import balance_feats
from src.features.haris.category_feats import category_feats
import pandas as pd
import time

def run_pipeline():
    # Load Data
    data_start = time.time()
    print("Loading data...")
    consDF, acctDF, trxnDF, cat_map = load_data()

    features_start = time.time()

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
        
    features[features.isna()] = 0 # Change to mean imputation

    df = consDF.merge(features, on='prism_consumer_id')
    
    print(f"Data loaded in {time.time() - data_start:.2f} seconds.")

    return df

if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())