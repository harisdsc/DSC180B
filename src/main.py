import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_clean_data
from src.features import calculate_running_balance, extract_ALL_features
from src.model import train_easy_ensemble
from src.evaluation import evaluate_and_plot
from src.logistics import train_logistic_ensemble
import shap

def prune_features_fast(X_train, y_train, max_features=100, corr_threshold=0.85):
    variances = X_train.var()
    X_train = X_train.drop(columns=variances[variances == 0].index)
    
    sample_df = X_train.sample(n=min(5000, len(X_train)), random_state=42) if len(X_train) > 5000 else X_train
    corr_matrix = sample_df.corr(method='spearman').abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    X_train_reduced = X_train.drop(columns=to_drop_corr)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_reduced, y_train)
    
    importance_df = pd.DataFrame({
        'feature': X_train_reduced.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df = importance_df[importance_df['importance'] > 0]
    top_features = importance_df.head(max_features)['feature'].tolist()
    
    return top_features

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
    y_train = train_df['DQ_TARGET']
    y_holdout = test_df['DQ_TARGET']

    print("🔪 Running advanced feature pruning...")
    top_feature_cols = prune_features_fast(
        X_train=X_train_full, 
        y_train=y_train, 
        max_features=150, 
        corr_threshold=0.85
    )
    print(f"✅ Final feature space size: {len(top_feature_cols)} columns.")
    train_df_pruned = train_df[['prism_consumer_id', 'DQ_TARGET'] + top_feature_cols]
    test_df_pruned = test_df[['prism_consumer_id', 'DQ_TARGET'] + top_feature_cols]

    print(f"Final feature space size: {len(top_feature_cols)} columns.")

    #best_threshold, final_test_probs = train_logistic_ensemble(train_df_pruned, test_df_pruned, num_iterations=20)
    best_threshold, final_test_probs, last_model = train_easy_ensemble(
        train_df_pruned, test_df_pruned, num_iterations=20
    )
    
    # 1. Isolate the pruned feature matrix for the holdout set
    X_holdout = test_df_pruned[top_feature_cols]
    
    # 2. Create the SHAP explainer using the model we just trained
    explainer = shap.TreeExplainer(last_model)
    
    # 3. Pass X_holdout and the explainer to generate all the plots
    evaluate_and_plot(X_holdout, y_holdout, final_test_probs, best_threshold, explainer=explainer)

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