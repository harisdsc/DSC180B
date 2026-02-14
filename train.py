from src.pipeline import run_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint, loguniform
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import time
import json
import sys

MODELS = ['log-reg', 'xgboost', 'catboost', 'lightgbm']

GPU_AVAILABLE = True if get_gpu_device_count() > 0 else False

def train_model(model_name, df, tune=False):
    model_start = time.time()

    X = df.drop(columns=['prism_consumer_id', 'DQ_TARGET', 'evaluation_date'])
    y = df['DQ_TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if model_name == 'log-reg':
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', LogisticRegression(max_iter=1000))
        ])
    elif model_name == 'xgboost':
        model = xgb.XGBClassifier(eval_metric='auc')
    elif model_name == 'catboost':
        model = CatBoostClassifier(verbose=0, allow_writing_files=False)
    elif model_name == 'lightgbm':
        model = lgb.LGBMClassifier(verbose=-1)
    else:
        print('Invalid model.')
        return 0

    if tune:
        tune_start = time.time()
        print(f"{time.strftime("%I:%M %p")} - Starting hyperparameter search for {model_name}...")
        param_dist = get_params(model_name)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        print(f"Best Params: {search.best_params_}")
        print(f"Best CV AUC: {search.best_score_}")
        print(f"Hyperparameter search time: {time.time() - tune_start} seconds.")
        model = search.best_estimator_

        with open(f"configs/{model_name}.json", "w") as f:
            json.dump(search.best_params_, f)
        
    else:
        print(f"Training {model_name} with default parameters...")
        model.fit(X_train, y_train)

    train_preds = model.predict_proba(X_train)[:, 1]
    train_score = roc_auc_score(y_train, train_preds)

    test_preds = model.predict_proba(X_test)[:, 1]
    test_score = roc_auc_score(y_test, test_preds)

    print(f'Training AUC: {train_score}')
    print(f'Testing AUC:  {test_score}')
    print(f"Total time: {time.time() - model_start} seconds.")

    return model

def get_params(model_name):
    if model_name == 'log-reg':
            return {
                'clf__C': loguniform(1e-4, 100),
                'clf__penalty': ['l2'],
                'clf__solver': ['lbfgs', 'liblinear']
            }
    elif model_name == 'xgboost':
        return {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5)
        }
    elif model_name == 'catboost':
        return {
            'iterations': randint(100, 1000),
            'depth': randint(4, 10),
            'learning_rate': loguniform(0.01, 0.3),
            'l2_leaf_reg': uniform(1, 10),
            'border_count': randint(32, 255)
        }
    elif model_name == 'lightgbm':
        return {
            'n_estimators': randint(100, 1000),
            'num_leaves': randint(20, 150),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_samples': randint(10, 100)
        }
    return {}

if __name__ == "__main__":
    total_start = time.time()
    args = sys.argv
    
    tune = '--tune' in args
    if tune:
        args.remove('--tune')
    
    df = run_pipeline()

    model = 'log-reg'
    if len(args) > 1:
        model = args[1]

    if model == 'all':
        print('-----------------------------')
        for model in MODELS:
            print(f'Processing {model}...')
            train_model(model, df, tune=tune)
            print('-----------------------------')
    else:
        train_model(model, df, tune=tune)

    print(f"Total time: {time.time() - total_start} seconds.")