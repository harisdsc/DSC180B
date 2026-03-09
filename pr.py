import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

from src.pipeline import run_pipeline
from train import prune_features_fast

def plot_overlaid_pr(model_folder='models'):
    # 1. Load the feature dataset
    df = run_pipeline()
    
    # 2. Replicate the exact split from the training script
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['DQ_TARGET'], random_state=111
    )
    
    # 3. Identify features using the same pruning logic
    X_train_full = train_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date'], errors='ignore')
    y_train = train_df['DQ_TARGET']
    top_features = prune_features_fast(X_train_full, y_train, max_features=50)
    
    X_test = test_df[top_features]
    y_test = test_df['DQ_TARGET']

    # 4. Initialize the plot
    plt.figure(figsize=(10, 8))
    
    # Calculate the no-skill baseline (proportion of the positive class)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

    # 5. Load and evaluate each .sav file
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.sav')]
    
    if not model_files:
        print(f"No .sav files found in {model_folder}")
        return

    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        model_name = model_file.replace('.sav', '').upper()
        
        print(f"Evaluating {model_name}...")
        model = joblib.load(model_path)
        
        # Get probabilities for the positive class (Delinquent)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
            
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{model_name} (PR AUC = {pr_auc:.3f})')

    # 6. Formatting the final plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overlaid Precision-Recall Curves: Delinquency Prediction')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig('overlaid_pr_comparison.png', dpi=300)
    plt.show()
    print("Graph saved as overlaid_pr_comparison.png")

if __name__ == "__main__":
    plot_overlaid_pr('models')