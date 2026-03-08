import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report, 
    roc_curve, precision_recall_curve, auc
)
import shap
import textwrap

# Import the mapping function from your reasons_code.py script
from src.reasons_code import generate_dynamic_reason

def evaluate_and_plot(X_holdout, y_holdout, final_test_probs, best_threshold, explainer=None, max_features_to_show=10, global_sample_size=1000):
    """
    Evaluates predictions, saves metrics plots at 300 DPI, generates a global SHAP feature 
    weight graph, creates a custom Top 20 feature importance plot, and saves filtered SHAP waterfall plots.
    """
    y_arr = np.array(y_holdout)
    probs_arr = np.array(final_test_probs)
    
    final_full_auc = roc_auc_score(y_arr, probs_arr)
    print(f"\n--- FINAL ENSEMBLE SUMMARY ---")
    print(f"Full Test ROC AUC: {final_full_auc:.4f}")

    roc_test_preds = (probs_arr >= best_threshold).astype(int)

    print(f"\nOptimal ROC Threshold (from training): {best_threshold:.4f}\n")
    print(classification_report(y_arr, roc_test_preds))

    # --- 1. PLOT METRICS (Hist, ROC, PR) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Probability Histogram
    axes[0].hist(probs_arr[y_arr == 0], bins=50, alpha=0.6, label='Actual Good (0)', color='green', density=True)
    axes[0].hist(probs_arr[y_arr == 1], bins=50, alpha=0.6, label='Actual Bad (1)', color='red', density=True)
    axes[0].axvline(best_threshold, color='black', linestyle='dashed', label=f'Threshold ({best_threshold:.3f})')
    axes[0].set_title('Test Set Predicted Probabilities')
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Panel B: ROC Curve
    fpr, tpr, _ = roc_curve(y_arr, probs_arr)
    axes[1].plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {final_full_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1].set_title('Receiver Operating Characteristic (ROC)')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc="lower right")

    # Panel C: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_arr, probs_arr)
    pr_auc = auc(recall, precision)
    axes[2].plot(recall, precision, color='purple', label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig('ensemble_metrics_results.png', dpi=300)
    plt.close()
    print("Metrics plots saved to ensemble_metrics_results.png (300 DPI)")

    # --- 2. SHAP EXPLANATIONS ---
    if explainer is not None:
        
        # --- A. Global Feature Weights Graph ---
        print("\nGenerating Global SHAP Feature Weights plot...")
        if len(X_holdout) > global_sample_size:
            X_sample = shap.utils.sample(X_holdout, global_sample_size)
            print(f"  (Sampling {global_sample_size} rows for global metrics)")
        else:
            X_sample = X_holdout
            
        shap_values_global = explainer(X_sample)
        
        # FAILSAFE & WRAPPER: Wrap long FCRA sentences to 45 characters max per line
        if hasattr(X_sample, 'columns'):
            readable_names = [textwrap.fill(generate_dynamic_reason(col), width=45) for col in X_sample.columns]
        else:
            readable_names = [textwrap.fill(generate_dynamic_reason(f"Feature {i}"), width=45) for i in range(X_sample.shape[1])]
            
        shap_values_global.feature_names = readable_names
        
        # Increase width slightly to accommodate longer sentence labels
        plt.figure(figsize=(14, 10))
        shap.plots.bar(shap_values_global, max_display=max_features_to_show, show=False)
        plt.title('Global Feature Weights (Mean Absolute SHAP)')
        plt.tight_layout()
        plt.savefig('shap_global_feature_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  -> Saved shap_global_feature_weights.png (300 DPI)")

        # --- B. Custom Top 20 Feature Importances Bar Chart ---
        print("\nGenerating Top 20 Overall Feature Importances plot...")
        
        if len(shap_values_global.values.shape) > 2:
            mean_abs_shap = np.abs(shap_values_global.values[:, :, 1]).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values_global.values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': readable_names,
            'Importance': mean_abs_shap
        })
        
        top_20_df = importance_df.sort_values(by='Importance', ascending=True).tail(20)
        
        plt.figure(figsize=(14, 12))
        plt.barh(top_20_df['Feature'], top_20_df['Importance'], height=0.6, color='#1f77b4')
        plt.title('Top 20 Overall Feature Importances', fontsize=16)
        plt.xlabel('Mean Absolute SHAP Value')
        plt.tight_layout()
        plt.savefig('top_20_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  -> Saved top_20_feature_importances.png (300 DPI)")

        # --- C. Local Waterfall Plots for TP & TN ---
        tp_indices = np.where((roc_test_preds == 1) & (y_arr == 1))[0]
        tn_indices = np.where((roc_test_preds == 0) & (y_arr == 0))[0]

        samples_to_plot = []
        for i in tp_indices[:2]: samples_to_plot.append((i, "TP"))
        for i in tn_indices[:2]: samples_to_plot.append((i, "TN"))

        if not samples_to_plot:
            print("Could not find enough TP/TN cases to plot SHAP values.")
            return

        print(f"\nGenerating filtered SHAP waterfall plots...")
        for idx, label in samples_to_plot:
            if hasattr(X_holdout, 'iloc'):
                row = X_holdout.iloc[[idx]]
            else:
                row = X_holdout[idx:idx+1]
            
            shap_values_local = explainer(row)

            # FAILSAFE & WRAPPER for Waterfall plots
            if hasattr(row, 'columns'):
                shap_values_local.feature_names = [textwrap.fill(generate_dynamic_reason(col), width=45) for col in row.columns]
            else:
                shap_values_local.feature_names = [textwrap.fill(generate_dynamic_reason(f"Feature {i}"), width=45) for i in range(row.shape[1])]

            plt.figure(figsize=(14, 9))
            shap.plots.waterfall(shap_values_local[0], max_display=max_features_to_show, show=False)
            
            plt.title(f'SHAP Waterfall - {label} (Index: {idx})')
            plt.tight_layout()
            
            filename = f'shap_waterfall_{label.lower()}_{idx}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved {filename} (300 DPI)")
            
    else:
        print("\nSkipping SHAP plots: No explainer object was provided.")