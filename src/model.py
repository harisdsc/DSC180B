import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from src.features import add_knn_error_features

def train_easy_ensemble(train_df, test_df, num_iterations=20):
    print("Initializing Pilot Model & Hard Negative Mining...")
    
    # Safely separate features from targets and identifiers
    X_train_full = train_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    y_train_full = train_df['DQ_TARGET']
    
    X_test_full = test_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')

    # --- 1. PILOT MODEL ---
    # n_jobs=-1 uses all CPU cores to make this run lightning fast
    pilot_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1)
    cv_preds = cross_val_predict(pilot_model, X_train_full, y_train_full, cv=5, method='predict_proba')[:, 1]
    
    # --- 2. DYNAMIC THRESHOLDING ---
    fpr, tpr, thresholds = roc_curve(y_train_full, cv_preds)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    # --- 3. K-NN FEATURE INJECTION & WEIGHTING ---
    X_train_knn, X_test_knn = add_knn_error_features(X_train_full.copy(), y_train_full, X_test_full.copy(), cv_preds)
    
    # Assign a 3x weight to consumers who look like defaults but aren't (Hard Negatives)
    sample_weights = np.where((y_train_full == 0) & (cv_preds > best_threshold), 3.0, 1.0)
    
    # --- 4. FAST ENSEMBLE TRAINING ---
    print(f"Training {num_iterations}-Model Balanced Ensemble...")
    ensemble_test_probs = np.zeros((len(X_test_knn), num_iterations))
    
    majority_ids = train_df[train_df['DQ_TARGET'] == 0]['prism_consumer_id'].values
    minority_ids = train_df[train_df['DQ_TARGET'] == 1]['prism_consumer_id'].values
    minority_count = len(minority_ids)
    
    for i in range(num_iterations):
        # Dynamically undersample the majority class
        sampled_maj = np.random.choice(majority_ids, size=minority_count, replace=False)
        under_ids = np.concatenate([sampled_maj, minority_ids])
        
        # Apply the mask to subset the data
        mask = train_df['prism_consumer_id'].isin(under_ids)
        X_sub = X_train_knn[mask]
        y_sub = y_train_full[mask]
        w_sub = sample_weights[mask] 
        
        # Train the individual tree
        xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, n_jobs=-1)
        xgb_model.fit(X_sub, y_sub, sample_weight=w_sub, verbose=False)
        
        # Log out-of-fold predictions
        ensemble_test_probs[:, i] = xgb_model.predict_proba(X_test_knn)[:, 1]
        
    # Average predictions across all 20 models
    final_test_probs = ensemble_test_probs.mean(axis=1)
    
    return 0.5, final_test_probs