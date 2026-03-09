import os
import time
import pandas as pd
from src.data_loader import load_and_clean_data
from src.features.feature_engineering import calculate_running_balance, extract_ALL_features

def run_pipeline():
    """
    Main execution function that loads raw data, computes advanced features,
    and caches the final dataset.
    """
    feature_cache_path = 'data/features.pqt'

    if os.path.exists(feature_cache_path):
        print(f"✅ Loading precomputed features from {feature_cache_path}...")
        df = pd.read_parquet(feature_cache_path)
        print(f"Total consumers loaded: {len(df):,}")
        return df

    print("🚀 Starting Advanced Feature Engineering Pipeline...")
    pipeline_start_time = time.time()

    data_directory = 'data'
    # load_and_clean_data handles filtering and merging
    consDF, acctDF, trxnDF = load_and_clean_data(data_path=data_directory)
    
    print("Calculating chronological running balances...")
    history_all = calculate_running_balance(trxnDF, acctDF)
    
    print("Extracting complex feature set (this may take several minutes)...")
    # extract_ALL_features handles all AI and statistical calculations
    master_features_df = extract_ALL_features(history_all, acctDF)
    
    master_features_df = master_features_df.dropna(subset=['DQ_TARGET'])
    master_features_df = master_features_df.groupby('prism_consumer_id').max().reset_index()
    
    print(f"Saving {len(master_features_df):,} consumers to {feature_cache_path}...") # Fix: Removed space after comma
    os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
    master_features_df.to_parquet(feature_cache_path)

    total_time = (time.time() - pipeline_start_time) / 60
    print(f"✅ Pipeline completed successfully in {total_time:.2f} minutes.")
    
    return master_features_df

if __name__ == "__main__":
    run_pipeline()