import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report

def evaluate_and_plot(y_holdout, final_test_probs, best_threshold):
    final_full_auc = roc_auc_score(y_holdout, final_test_probs)
    print(f"\n--- FINAL ENSEMBLE SUMMARY ---")
    print(f"Full Test ROC AUC: {final_full_auc:.4f}")

    roc_test_preds = (final_test_probs >= best_threshold).astype(int)

    print(f"\nOptimal ROC Threshold (from training): {best_threshold:.4f}\n")
    print(classification_report(y_holdout, roc_test_preds))

    plt.figure(figsize=(10, 6))
    plt.hist(final_test_probs[y_holdout == 0], bins=50, alpha=0.6, label='Actual Good (0)', color='green', density=True)
    plt.hist(final_test_probs[y_holdout == 1], bins=50, alpha=0.6, label='Actual Bad (1)', color='red', density=True)
    plt.axvline(best_threshold, color='black', linestyle='dashed', label=f'Threshold ({best_threshold:.3f})')
    plt.title('Test Set Predicted Probabilities')
    plt.legend()
    plt.savefig('ensemble_results.png')
    print("Plot saved to ensemble_results.png")