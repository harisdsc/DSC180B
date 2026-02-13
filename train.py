from src.pipeline import run_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import time
import sys

def train_model(model, df):

    model_start = time.time()

    X = df.drop(columns=['prism_consumer_id', 'DQ_TARGET', 'evaluation_date'])
    y = df['DQ_TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model == 'log-reg':
        print('Training Logistic Regression model...')
        model = LogisticRegression()
        model.fit(X_train, y_train)
    elif model == 'xgboost':
        print('Training XGBoost model...')
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
    elif model == 'catboost':
        print('Training CatBoost model...')
        model = CatBoostClassifier(verbose=0)
        model.fit(X_train, y_train)
    elif model =='lightgbm':
        print('Training LightGBM model...')
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
    else:
        print('Invalid model.')
        return 0
    
    train_preds = model.predict_proba(X_train)[:, 1]
    train_score = roc_auc_score(y_train, train_preds)

    test_preds = model.predict_proba(X_test)[:, 1]
    test_score = roc_auc_score(y_test, test_preds)

    print(f'Training AUC: {train_score}')
    print(f'Testing AUC: {test_score}')

    print(f"Model trained in {time.time() - model_start:.2f} seconds.")

    return model

if __name__ == "__main__":
    total_start = time.time()
    args = sys.argv
    model = 'log-reg'
    df = run_pipeline()
    if len(args) > 1:
        if args[1] == 'all':
            print('-----------------------------')
            for model in ['log-reg', 'xgboost', 'catboost', 'lightgbm']:
                print(f'{model}:')
                train_model(model, df)
                print('-----------------------------')
            print(f"Total time: {time.time() - total_start:.2f} seconds.")
            sys.exit(0)
        else:
            model = args[1]
    train_model(model, df)
    print(f"Total time: {time.time() - total_start:.2f} seconds.")
