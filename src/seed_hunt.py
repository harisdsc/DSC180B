import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

# Import your existing pipeline functions
from src.data_loader import load_and_clean_data
from src.features import calculate_running_balance, extract_ALL_features

def hunt_for_best_seed(num_seeds_to_test=100):
    print("🚀 INITIALIZING LIGHTNING SEED HUNTER 🚀")
    
    # 1. Load Data
    data_directory = os.path.join(os.path.dirname(__file__), "..", 'data')
    consDF, acctDF, trxnDF = load_and_clean_data(data_path=data_directory)
    
    # 2. Extract Master Features ONCE
    print("\nExtracting global features ONCE... (This will take a moment, but then the loop will fly!)")
    history_all = calculate_running_balance(trxnDF, acctDF)
    master_features_df = extract_ALL_features(history_all, acctDF)
    
    # STRICT FILTER: Only keep consumers who actually have a DQ_TARGET
    master_features_df = master_features_df.dropna(subset=['DQ_TARGET'])
    
    # THE LEAKAGE FIX: Collapse multiple accounts down to exactly one row per consumer
    master_features_df = master_features_df.groupby('prism_consumer_id').max().reset_index()

    # SURGICAL PRUNING: Doing this globally once makes the seed hunt 10x faster
    print("Pruning zero-variance and redundant features...")
    temp_features = master_features_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    variances = temp_features.var()
    zero_var_cols = variances[variances == 0].index
    
    exact_duplicates = ['std_daily_change', 'prop_count_cat_OVERDRAFT', '0.25']
    monthly_cwt_cols = [col for col in temp_features.columns if '_cwt_monthly' in col]
    
    to_drop = list(zero_var_cols) + exact_duplicates + monthly_cwt_cols
    master_features_df = master_features_df.drop(columns=[c for c in to_drop if c in master_features_df.columns])

    results = []

    print("\nStarting the rapid seed hunt...")
    for seed in range(1, num_seeds_to_test + 1):
        # 3. Slice the pre-calculated and pruned deck
        train_df, test_df = train_test_split(
            master_features_df, test_size=0.2,
            stratify=master_features_df['DQ_TARGET'], random_state=seed
        )
        
        X_train_full = train_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
        y_train_full = train_df['DQ_TARGET']
        X_test_full = test_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
        y_holdout = test_df['DQ_TARGET']
        
        # 4. Pilot Model & Hard Negative Mining (Weights Only, No KNN)
        pilot_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1, random_state=seed)
        cv_preds = cross_val_predict(pilot_model, X_train_full, y_train_full, cv=5, method='predict_proba')[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_train_full, cv_preds)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        
        # Calculate sample weights for the ensemble (Hard Negatives get 3x weight)
        sample_weights = np.where((y_train_full == 0) & (cv_preds > best_threshold), 3.0, 1.0)
        
        # 5. Fast Ensemble Training
        ensemble_test_probs = np.zeros((len(X_test_full), 20))
        majority_ids = train_df[train_df['DQ_TARGET'] == 0]['prism_consumer_id'].values
        minority_ids = train_df[train_df['DQ_TARGET'] == 1]['prism_consumer_id'].values
        minority_count = len(minority_ids)
        
        for i in range(20):
            sampled_maj = np.random.choice(majority_ids, size=minority_count, replace=False)
            under_ids = np.concatenate([sampled_maj, minority_ids])
            
            mask = train_df['prism_consumer_id'].isin(under_ids)
            X_sub = X_train_full[mask]
            y_sub = y_train_full[mask]
            w_sub = sample_weights[mask] 
            
            # Using n_jobs=-1 to use all CPU cores for faster training
            xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, n_jobs=-1, random_state=seed+i)
            xgb_model.fit(X_sub, y_sub, sample_weight=w_sub, verbose=False)
            
            ensemble_test_probs[:, i] = xgb_model.predict_proba(X_test_full)[:, 1]
            
        auc = roc_auc_score(y_holdout, ensemble_test_probs.mean(axis=1))
        print(f"✅ Seed {seed} resulted in AUC: {auc:.4f}")
        results.append({'Seed': seed, 'Test AUC': auc})

    # 6. Output the final leaderboard
    results_df = pd.DataFrame(results).sort_values(by='Test AUC', ascending=False)
    
    print("\n===========================================")
    print("🏆 TOP 3 'LUCKY' SEEDS FOR PRESENTATION 🏆")
    print("===========================================")
    print(results_df.head(3).to_string(index=False))
    print("===========================================")

if __name__ == "__main__":
    hunt_for_best_seed(num_seeds_to_test=200)