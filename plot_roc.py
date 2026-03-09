import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from src.pipeline import run_pipeline
from sklearn.model_selection import train_test_split


# Color map for consistent, distinct colors per model type
MODEL_COLORS = {
    'xgboost': '#2196F3',
    'xgboost_tuned': '#0D47A1',
    'lightgbm': '#4CAF50',
    'lightgbm_tuned': '#1B5E20',
    'catboost': '#FF9800',
    'catboost_tuned': '#E65100',
    'log-reg': '#9C27B0',
    'log-reg_tuned': '#4A148C',
}


if __name__ == "__main__":
    # Load data with same split used during training
    df = run_pipeline()
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['DQ_TARGET'], random_state=363)
    y_test = test_df['DQ_TARGET']

    # Discover all saved models
    model_dir = 'models'
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.sav')])

    if not model_files:
        print("❌ No .sav model files found in models/ directory.")
        exit(1)

    fig, ax = plt.subplots(figsize=(10, 8))
    results = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace('.sav', '')
        features_path = os.path.join(model_dir, f"{model_name}_features.json")

        if not os.path.exists(features_path):
            print(f"⚠️  Skipping {model_name} — no {model_name}_features.json found.")
            continue

        try:
            # Load feature list
            with open(features_path, 'r') as f:
                feature_cols = json.load(f)

            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Predict
            X_test = test_df[feature_cols]
            probs = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, probs)
            fpr, tpr, _ = roc_curve(y_test, probs)

            # Pick color
            color = MODEL_COLORS.get(model_name, None)
            label = f"{model_name} (AUC = {auc:.4f})"

            ax.plot(fpr, tpr, label=label, linewidth=2.2, color=color)
            results.append({'model': model_name, 'auc': auc})
            print(f"✅ {model_name}: AUC = {auc:.4f}")

        except Exception as e:
            print(f"❌ Failed to process {model_file}: {e}")

    if not results:
        print("❌ No models were successfully loaded.")
        exit(1)

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='Random (AUC = 0.5)')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curves — All Trained Models', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.25)
    fig.tight_layout()

    # Save
    os.makedirs('images', exist_ok=True)
    save_path = 'images/roc_curves.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to {save_path}")

    # Print summary table
    results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
    print("\n" + "=" * 40)
    print("       MODEL RANKING BY AUC")
    print("=" * 40)
    for i, row in results_df.iterrows():
        print(f"  {row['model']:<22s} {row['auc']:.4f}")
    print("=" * 40)



    plt.show()