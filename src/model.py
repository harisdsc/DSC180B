import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve

def train_easy_ensemble(history_train, train_acct, test_features_df, num_iterations=20):
    from src.features import extract_ALL_features, add_knn_error_features
    
    X_holdout_base = test_features_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    print("Extracting all train features (this may take a moment)...")
    all_train_features_df = extract_ALL_features(history_train, train_acct)
    
    X_train_full = all_train_features_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    y_train_full = all_train_features_df['DQ_TARGET']
    X_holdout_full = X_holdout_base.copy()

    pilot_model = xgb.XGBClassifier(n_estimators=100, max_depth=3)
    
    print("Generating out-of-fold predictions for KNN hard negatives...")
    # Get out-of-fold predictions to prevent in-sample leakage
    cv_preds = cross_val_predict(pilot_model, X_train_full, y_train_full, cv=5, method='predict_proba')[:, 1]
    
    # Calculate optimal threshold cleanly on training data
    fpr, tpr, thresholds = roc_curve(y_train_full, cv_preds)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    # Pass the numerical probabilities (cv_preds) instead of the pilot_model!
    X_train_full, X_holdout_full = add_knn_error_features(X_train_full, y_train_full, X_holdout_full, cv_preds)

    # Calculate weights dynamically based on the best threshold
    sample_weights = np.where((y_train_full == 0) & (cv_preds > best_threshold), 3.0, 1.0)

    ensemble_test_probs = np.zeros((len(X_holdout_full), num_iterations))
    
    majority_ids = all_train_features_df[all_train_features_df['DQ_TARGET'] == 0]['prism_consumer_id'].values
    minority_ids = all_train_features_df[all_train_features_df['DQ_TARGET'] == 1]['prism_consumer_id'].values
    minority_count = len(minority_ids)
    
    print("Training the weighted ensemble...")
    for i in range(num_iterations):
        # Balanced sampling
        sampled_maj = np.random.choice(majority_ids, size=minority_count, replace=False)
        under_ids = np.concatenate([sampled_maj, minority_ids])
        
        # Get indices for weights
        mask = all_train_features_df['prism_consumer_id'].isin(under_ids)
        X_sub = X_train_full[mask]
        y_sub = y_train_full[mask]
        w_sub = sample_weights[mask] 

        xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3)
        
        # Fit with weights
        xgb_model.fit(X_sub, y_sub, sample_weight=w_sub, verbose=False)
        
        ensemble_test_probs[:, i] = xgb_model.predict_proba(X_holdout_full)[:, 1]

    return 0.5, ensemble_test_probs.mean(axis=1)