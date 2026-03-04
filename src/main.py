import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_clean_data
from src.features import calculate_running_balance, extract_ALL_features
from src.model import train_easy_ensemble
from src.evaluation import evaluate_and_plot

def main():
    pipeline_start_time = time.time()
    print("🚀 Starting Credit Risk Pipeline...")

    data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
    consDF, acctDF, trxnDF = load_and_clean_data(data_path=data_directory)
    
    history_all = calculate_running_balance(trxnDF, acctDF)
    master_features_df = extract_ALL_features(history_all, acctDF)
    
    master_features_df = master_features_df.dropna(subset=['DQ_TARGET'])
    master_features_df = master_features_df.groupby('prism_consumer_id').max().reset_index()
    total_consumers = len(master_features_df)
    
    train_df, test_df = train_test_split(
        master_features_df, test_size=0.2,
        stratify=master_features_df['DQ_TARGET'],
        random_state= 111
    )
    
    X_train_full = train_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    X_test_full = test_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')

    variances = X_train_full.var()
    zero_var_cols = variances[variances == 0].index
    
    exact_duplicates = [
        'std_daily_change', 
        'prop_count_cat_OVERDRAFT',
        '0.25'
    ]
    
    monthly_cwt_cols = [col for col in X_train_full.columns if '_cwt_monthly' in col]
    
    to_drop = list(zero_var_cols) + exact_duplicates + monthly_cwt_cols
    
    train_df_pruned = train_df.drop(columns=[c for c in to_drop if c in train_df.columns])
    test_df_pruned = test_df.drop(columns=[c for c in to_drop if c in test_df.columns])

    print(f"Final feature space size: {train_df_pruned.shape[1]} columns.")
    y_holdout = test_df_pruned['DQ_TARGET']

    best_threshold, final_test_probs = train_easy_ensemble(
        train_df_pruned, test_df_pruned, num_iterations=20
    )
    evaluate_and_plot(y_holdout, final_test_probs, best_threshold)

    pipeline_end_time = time.time()
    total_time_seconds = pipeline_end_time - pipeline_start_time
    total_time_minutes = total_time_seconds / 60
    time_per_consumer = total_time_seconds / total_consumers

    print("\n==================================================")
    print("⏱️  PIPELINE LATENCY REPORT")
    print("==================================================")
    print(f"Total Labeled Consumers   : {total_consumers:,}")
    print(f"Total Pipeline Time       : {total_time_minutes:.2f} minutes ({total_time_seconds:.1f} seconds)")
    print(f"Average Time per Consumer : {time_per_consumer:.4f} seconds")
    print("==================================================\n")

if __name__ == "__main__":
    main()