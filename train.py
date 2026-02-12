from src.pipeline import run_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import sys

def train_model(model):
    df = run_pipeline()
    X = df.drop(columns=['prism_consumer_id', 'DQ_TARGET', 'evaluation_date'])
    y = df['DQ_TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model == 'log-reg':
        model = LogisticRegression()
        model.fit(X_train, y_train)

    elif model == 'xgboost':
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

    elif model == 'catboost':
        print('Training CatBoost model...')
        model = CatBoostClassifier(verbose=0)
        model.fit(X_train, y_train)

    else:
        print('Invalid model choice.')
        return 0
    
    train_preds = model.predict_proba(X_train)[:, 1]
    train_score = roc_auc_score(y_train, train_preds)

    test_preds = model.predict_proba(X_test)[:, 1]
    test_score = roc_auc_score(y_test, test_preds)

    print(f"Training AUC: {train_score}")
    print(f"Testing AUC: {test_score}")
    return model

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        model = args[1]
    else:
        model = 'log-reg'
    train_model(model)
    
