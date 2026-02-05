from pipeline import run_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd

def train_model():
    df = run_pipeline()
    X = df.drop(columns=['prism_consumer_id', 'DQ_TARGET', 'evaluation_date'])
    y = df['DQ_TARGET']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    preds = model.predict_proba(X)[:, 1]
    
    score = roc_auc_score(y, preds)

    print(f"Training AUC: {score}")
    return model

if __name__ == "__main__":
    train_model()
    
