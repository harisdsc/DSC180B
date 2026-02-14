from src.pipeline import run_pipeline
from src.tuning import tune_with_optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
import pandas as pd
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
    scores = pd.DataFrame(columns=['model', 'train_auc', 'test_auc', 'train_time', 'tune_time', 'date'])

    if tune and model_name in MODELS:
        tune_start = time.time()
        model, best_params = tune_with_optuna(model_name, X_train, y_train, n_trials=30)

        with open(f"configs/{model_name}.json", "w") as f:
            json.dump(best_params, f, indent=4)

        print(f"Hyperparameter search time: {time.time() - tune_start:.2f} seconds.")
        
    else:
        print(f"Training {model_name} with default parameters...")

        if model_name == 'log-reg':
            model = Pipeline([
                ('scaler', StandardScaler()), 
                ('clf', LogisticRegression(max_iter=25_000, class_weight='balanced'))
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

    model.fit(X_train, y_train)

    train_preds = model.predict_proba(X_train)[:, 1]
    train_score = roc_auc_score(y_train, train_preds)

    test_preds = model.predict_proba(X_test)[:, 1]
    test_score = roc_auc_score(y_test, test_preds)

    print(f'Training AUC: {train_score}')
    print(f'Testing AUC:  {test_score}')
    print(f"Total time: {time.time() - model_start:.2f} seconds.")

    scores = scores.append({'model': model_name, 'train_auc': train_score, 
                            'test_auc': test_score, 'train_time': time.time() - model_start, 
                            'tune_time': time.time() - tune_start if tune else 0,
                            'date': pd.Timestamp.now()}, ignore_index=True)
    scores.to_csv('model_scores.csv', index=False)

    return model

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
        for m in MODELS:
            print(f'Processing {m}...')
            train_model(m, df, tune=tune)
            print('-----------------------------')
    else:
        train_model(model, df, tune=tune)

    print(f"Total Execution time: {time.time() - total_start:.2f} seconds.")