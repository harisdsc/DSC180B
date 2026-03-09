from src.pipeline import run_pipeline
from src.models.ensemble import train_ensemble
from src.models.tuning import tune_hyperparameters
from src.evaluation import evaluate_and_plot
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from datetime import datetime
from sklearn.model_selection import train_test_split

import sys
import os
import json
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle


def prune_features_fast(X_train, y_train, max_features=50, corr_threshold=0.85):
    print("🔪 Running advanced feature selection...")
    variances = X_train.var()
    X_train = X_train.drop(columns=variances[variances == 0].index)
    
    sample_df = X_train.sample(n=min(5000, len(X_train)), random_state=42) if len(X_train) > 5000 else X_train
    corr_matrix = sample_df.corr(method='spearman').abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    X_train_reduced = X_train.drop(columns=to_drop_corr)
    
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_reduced, y_train)
    
    importance_df = pd.DataFrame({
        'feature': X_train_reduced.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df[importance_df['importance'] > 0].head(max_features)['feature'].tolist()
    print(f"✅ Final feature space size: {len(top_features)} columns.")
    return top_features


if __name__ == "__main__":
    total_start = time.time()
    args = sys.argv
    
    # Parse Flag
    tune = '--tune' in args
    if tune: args.remove('--tune')
    
    # Model selection (default to xgboost)
    model_name = args[1] if len(args) > 1 else 'xgboost'
    valid_models = ['xgboost', 'lightgbm', 'catboost', 'log-reg']
    if model_name not in valid_models:
        print(f"Invalid model. Choose from {valid_models}. Defaulting to xgboost.")
        model_name = 'xgboost'
    
    # 1. Load Data
    df = run_pipeline()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['DQ_TARGET'], random_state=363)
    
    X_train_full = train_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    y_train = train_df['DQ_TARGET']
    y_holdout = test_df['DQ_TARGET']

    # 2. Prune Features
    top_feature_cols = prune_features_fast(X_train_full, y_train, max_features=50)
    train_df_pruned = train_df[['prism_consumer_id', 'DQ_TARGET'] + top_feature_cols]
    test_df_pruned = test_df[['prism_consumer_id', 'DQ_TARGET'] + top_feature_cols]
    X_holdout = test_df[top_feature_cols]

    # 3. Optional Tuning
    best_params = {}
    tune_latency = 0.0
    if tune:
        tune_start = time.time()
        best_params, _ = tune_hyperparameters(model_name, train_df[top_feature_cols], y_train, n_trials=1000)
        tune_latency = time.time() - tune_start
        os.makedirs('configs', exist_ok=True)
        with open(f"configs/{model_name}.json", "w") as f:
            json.dump(best_params, f, indent=4)

    # 4. Train (Always Ensemble)
    train_start = time.time()
    best_threshold, final_test_probs, last_model = train_ensemble(
        model_name=model_name,
        train_df=train_df_pruned, 
        test_df=test_df_pruned, 
        num_iterations=20,
        best_params=best_params
    )
    train_latency = time.time() - train_start

    # 5. Classification Report & ROC AUC
    y_pred = (final_test_probs >= best_threshold).astype(int)
    
    print("\n" + "=" * 60)
    print("📊 Classification Report")
    print("=" * 60)
    print(classification_report(y_holdout, y_pred, digits=4))
    
    roc_auc = roc_auc_score(y_holdout, final_test_probs)
    acc = accuracy_score(y_holdout, y_pred)
    prec = precision_score(y_holdout, y_pred, zero_division=0)
    rec = recall_score(y_holdout, y_pred, zero_division=0)
    f1 = f1_score(y_holdout, y_pred, zero_division=0)

    # Compute train AUC
    X_train_pruned = train_df[top_feature_cols]
    train_probs = last_model.predict_proba(X_train_pruned)[:, 1]
    train_roc_auc = roc_auc_score(y_train, train_probs)

    print(f"🎯 Train ROC AUC Score: {train_roc_auc:.4f}")
    print(f"🎯 Test ROC AUC Score: {roc_auc:.4f}")
    print(f"🔑 Best Threshold: {best_threshold:.4f}")
    print("=" * 60)

    # 6. Save trained model
    os.makedirs('models', exist_ok=True)
    suffix = "_tuned" if tune else ""
    model_filename = f"models/{model_name}{suffix}.sav"
    features_filename = f"models/{model_name}{suffix}_features.json"
    
    pickle.dump(last_model, open(model_filename, 'wb'))
    with open(features_filename, 'w') as f:
        json.dump(top_feature_cols, f)
    
    print(f"\n💾 Model saved to {model_filename}")
    print(f"📋 Feature list saved to {features_filename}")

    # 7. Log run metadata to CSV
    total_latency = time.time() - total_start
    os.makedirs('logs', exist_ok=True)
    run_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': model_name,
        'tuned': tune,
        'best_params': json.dumps(best_params) if best_params else '{}',
        'num_features': len(top_feature_cols),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_pos_rate': float(y_train.mean()),
        'test_pos_rate': float(y_holdout.mean()),
        'best_threshold': round(best_threshold, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'train_roc_auc': round(train_roc_auc, 4),
        'roc_auc': round(roc_auc, 4),
        'tune_latency_sec': round(tune_latency, 2),
        'train_latency_sec': round(train_latency, 2),
        'total_latency_sec': round(total_latency, 2),
        'model_file': model_filename,
    }
    pd.DataFrame([run_info]).to_csv('data/models.csv', mode='a', header=False, index=False)

    # 8. SHAP & Plotting
    # print("Generating SHAP Explanations...")
    # if model_name == 'log-reg':
    #     imputer = last_model.named_steps['imputer']
    #     scaler = last_model.named_steps['scaler']
    #     X_holdout_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_holdout)), columns=X_holdout.columns)
    #     explainer = shap.LinearExplainer(last_model.named_steps['clf'], X_holdout_scaled)
    #     evaluate_and_plot(X_holdout_scaled, y_holdout, final_test_probs, best_threshold, explainer=explainer)
    # else:
    #     explainer = shap.TreeExplainer(last_model)
    #     evaluate_and_plot(X_holdout, y_holdout, final_test_probs, best_threshold, explainer=explainer)

    print(f"\nExecution complete: {total_latency/60:.2f} minutes.")