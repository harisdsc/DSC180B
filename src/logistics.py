import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors

def train_logistic_ensemble(train_df, test_df, num_iterations=20):
    print("Initializing Logistic Pilot Model & Preprocessing...")
    
    X_train_raw = train_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    y_train_full = train_df['DQ_TARGET'].values
    
    X_test_raw = test_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_full = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_test_full = scaler.transform(imputer.transform(X_test_raw))

    pilot_model = LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1)
    cv_preds = cross_val_predict(pilot_model, X_train_full, y_train_full, cv=5, method='predict_proba')[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_train_full, cv_preds)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    sample_weights = np.where((y_train_full == 0) & (cv_preds > best_threshold), 3.0, 1.0)
    
    print("Computing distance to hard negatives (KNN)...")
    hn_mask = (y_train_full == 0) & (cv_preds > best_threshold)
    hard_negatives = X_train_full[hn_mask]
    
    if len(hard_negatives) > 0:
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(hard_negatives)
        
        dist_train = knn.kneighbors(X_train_full)[0].mean(axis=1, keepdims=True)
        dist_test = knn.kneighbors(X_test_full)[0].mean(axis=1, keepdims=True)
    else:
        dist_train = np.zeros((X_train_full.shape[0], 1))
        dist_test = np.zeros((X_test_full.shape[0], 1))
        
    X_train_knn = np.hstack((X_train_full, dist_train))
    X_test_knn = np.hstack((X_test_full, dist_test))
    
    print(f"Training {num_iterations}-Model Balanced Logistic Ensemble...")
    ensemble_test_probs = np.zeros((len(X_test_knn), num_iterations))
    
    majority_indices = np.where(y_train_full == 0)[0]
    minority_indices = np.where(y_train_full == 1)[0]
    minority_count = len(minority_indices)
    
    for i in range(num_iterations):
        sampled_maj = np.random.choice(majority_indices, size=minority_count, replace=False)
        under_indices = np.concatenate([sampled_maj, minority_indices])
        
        X_sub = X_train_knn[under_indices]
        y_sub = y_train_full[under_indices]
        w_sub = sample_weights[under_indices] 
        
        lr_model = LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=111+i)
        lr_model.fit(X_sub, y_sub, sample_weight=w_sub)
        
        ensemble_test_probs[:, i] = lr_model.predict_proba(X_test_knn)[:, 1]
        
    final_test_probs = ensemble_test_probs.mean(axis=1)
    
    return 0.5, final_test_probs