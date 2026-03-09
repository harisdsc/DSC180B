import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report, 
    roc_curve, precision_recall_curve, auc
)
import shap

def clean_feature_name_for_plot(feat):
    """
    Translates raw feature column names into clean, short, plain English titles 
    for data visualization (instead of long FCRA sentences).
    """
    # 1. Exact matches for standard features
    exact_mapping = {
        'avg_balance': 'Average Account Balance',
        'min_balance': 'Minimum Account Balance',
        'max_balance': 'Maximum Account Balance',
        'checking_to_savings_ratio': 'Checking to Savings Ratio',
        'txn_rate_to_balance_ratio': 'Transaction Rate to Balance Ratio',
        'avg_monthly_income': 'Average Monthly Income',
        'spending_volatility': 'Spending Volatility',
        'liquidity_stress_ratio': 'Liquidity Stress Ratio',
        'avg_txn_freq_per_month': 'Avg Monthly Transaction Frequency',
        'num_accounts': 'Number of Active Accounts',
        'total_outflows': 'Total Monthly Outflows',
        'wealth_transfer_ratio': 'Wealth Transfer Ratio',
        'stress_outflow_ratio': 'Stress Outflow Ratio',
        'cash_utilization_pct': 'Cash Utilization Pct',
        'BNPL_utilization_pct': 'BNPL Utilization Pct',
        'mean_daily_change': 'Mean Daily Balance Change',
        'std_daily_change': 'Balance Volatility (Std Dev)',
        '0.25': '25th Percentile Balance',
        '0.5': 'Median Historical Balance',
        '0.75': '75th Percentile Balance',
        'dist_to_hard_neg': 'Distance to Hard Negative Profile',
    }
    
    # AI Engineered Features
    ai_mapping = {
        'recovery_latency_normalized': 'Normalized Recovery Latency',
        'balance_autocorr_lag1_index': 'Balance Autocorrelation (Lag 1)',
        'balance_dist_skewness_index': 'Balance Distribution Skewness',
        'balance_path_efficiency_index': 'Balance Path Efficiency',
        'asymmetry_recovery_flow_index': 'Asymmetry Recovery Flow',
        'volatility_concentration_ratio': 'Volatility Concentration Ratio',
        'nonlinear_transformed_ratio': 'Non-linear Balance Ratio',
        'stress_intensity_asset_ratio': 'Stress Intensity Asset Ratio',
        'novel_balance_state_switching_volatility': 'Balance State Switching Volatility',
        'recent_volatility_regime_shift_index': 'Recent Volatility Regime Shift',
        'balance_spending_sensitivity_index': 'Balance Spending Sensitivity',
    }
    
    exact_mapping.update(ai_mapping)
    
    if str(feat) in exact_mapping:
        return exact_mapping[str(feat)]
        
    # 2. Dynamic Regex/String Parsing for Categories
    name = str(feat)
    
    # Wavelet Weekly/Monthly Patterns
    if name.startswith('cat_') and name.endswith('_cwt_weekly'):
        cat = name.replace('cat_', '').replace('_cwt_weekly', '').replace('_', ' ').title()
        return f"Weekly {cat} Pattern"
        
    if name.startswith('cat_') and name.endswith('_cwt_monthly'):
        cat = name.replace('cat_', '').replace('_cwt_monthly', '').replace('_', ' ').title()
        return f"Monthly {cat} Pattern"
        
    # Proportional Counts
    if name.startswith('prop_count_cat_'):
        cat = name.replace('prop_count_cat_', '').replace('_', ' ').title()
        return f"Proportion of {cat} Txns"
        
    # Day of Month Averages
    if name.startswith('cat_') and '_dom_' in name:
        parts = name.split('_dom_')
        cat = parts[0].replace('cat_', '').replace('_', ' ').title()
        day = parts[1]
        return f"{cat} Avg (Day {day})"
        
    # Day of Week Averages
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for day in days:
        if name.startswith('cat_') and name.endswith(f'_{day}'):
            cat = name.replace('cat_', '').replace(f'_{day}', '').replace('_', ' ').title()
            return f"{cat} Avg ({day.title()})"
            
    # 3. Ultimate Fallback (Replaces underscores with spaces and capitalizes)
    return name.replace('_', ' ').title()


def evaluate_and_plot(X_holdout, y_holdout, final_test_probs, best_threshold, explainer=None, max_features_to_show=10, global_sample_size=1000):
    """
    Evaluates predictions, saves metrics plots at 300 DPI, generates a global SHAP feature 
    weight graph, a classic SHAP beeswarm summary plot, and filtered SHAP waterfall plots.
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
        
        print("\nGenerating Global SHAP samples...")
        if len(X_holdout) > global_sample_size:
            X_sample = shap.utils.sample(X_holdout, global_sample_size)
            print(f"  (Sampling {global_sample_size} rows for global metrics)")
        else:
            X_sample = X_holdout
            
        shap_values_global = explainer(X_sample)
        
        # Format the feature names to be clean and punchy
        if hasattr(X_sample, 'columns'):
            readable_names = [clean_feature_name_for_plot(col) for col in X_sample.columns]
        else:
            readable_names = [clean_feature_name_for_plot(f"Feature_{i}") for i in range(X_sample.shape[1])]
            
        shap_values_global.feature_names = readable_names
        
        # --- A. Classic SHAP Summary Plot (Beeswarm) ---
        print("Generating Classic SHAP Summary (Beeswarm) plot...")
        plt.figure(figsize=(10, 8))
        
        # Depending on XGBoost objective, SHAP returns 2D or 3D arrays. 
        # If it's a classification model it often returns 3D: (samples, features, classes)
        if len(shap_values_global.values.shape) > 2:
            shap_vals_for_plot = shap_values_global.values[:, :, 1]
        else:
            shap_vals_for_plot = shap_values_global.values
            
        shap.summary_plot(
            shap_vals_for_plot, 
            X_sample, 
            feature_names=readable_names, 
            max_display=max_features_to_show, 
            show=False
        )
        plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  -> Saved shap_summary_beeswarm.png (300 DPI)")

        # --- B. Global Feature Weights Graph (Bar) ---
        print("Generating Global SHAP Feature Weights (Bar) plot...")
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values_global, max_display=max_features_to_show, show=False)
        plt.title('Global Feature Weights (Mean Absolute SHAP)')
        plt.tight_layout()
        plt.savefig('shap_global_feature_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  -> Saved shap_global_feature_weights.png (300 DPI)")

        # --- C. Local Waterfall Plots for TP & TN ---
        tp_indices = np.where((roc_test_preds == 1) & (y_arr == 1))[0]
        tn_indices = np.where((roc_test_preds == 0) & (y_arr == 0))[0]

        samples_to_plot = []
        for i in tp_indices[:2]: samples_to_plot.append((i, "TP"))
        for i in tn_indices[:2]: samples_to_plot.append((i, "TN"))

        if not samples_to_plot:
            print("Could not find enough TP/TN cases to plot SHAP values.")
            return

        print(f"Generating filtered SHAP waterfall plots...")
        for idx, label in samples_to_plot:
            if hasattr(X_holdout, 'iloc'):
                row = X_holdout.iloc[[idx]]
            else:
                row = X_holdout[idx:idx+1]
            
            shap_values_local = explainer(row)

            # Apply clean names to waterfall plots
            if hasattr(row, 'columns'):
                shap_values_local.feature_names = [clean_feature_name_for_plot(col) for col in row.columns]
            else:
                shap_values_local.feature_names = [clean_feature_name_for_plot(f"Feature_{i}") for i in range(row.shape[1])]

            plt.figure(figsize=(10, 7))
            # Fallback to bar if waterfall has issues with versions
            try:
                shap.plots.waterfall(shap_values_local[0], max_display=max_features_to_show, show=False)
            except Exception:
                shap.plots.bar(shap_values_local[0], max_display=max_features_to_show, show=False)
            
            plt.title(f'SHAP Waterfall - {label} (Index: {idx})')
            plt.tight_layout()
            
            filename = f'shap_waterfall_{label.lower()}_{idx}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved {filename} (300 DPI)")
            
    else:
        print("\nSkipping SHAP plots: No explainer object was provided.")