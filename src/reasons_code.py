import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from features import calculate_running_balance, extract_ALL_features

FCRA_MAPPING = {
    'avg_balance': 'Average account balance is critically low.',
    'min_balance': 'Account balance dropped to insufficient levels.',
    'max_balance': 'Peak account balances are insufficient to support outflow.',
    'stress_outflow_ratio': 'High proportion of stress-related outflows (e.g., overdrafts, fees).',
    'wealth_transfer_ratio': 'Insufficient wealth accumulation or investment transfers.',
    'BNPL_utilization_pct': 'High reliance on short-term Buy-Now-Pay-Later (BNPL) debt.',
    'cash_utilization_pct': 'High reliance on ATM cash withdrawals relative to electronic payments.',
    'checking_to_savings_ratio': 'Low savings liquidity relative to checking outflows.',
    'txn_rate_to_balance_ratio': 'High daily transaction volume relative to available liquidity.',
    'avg_monthly_income': 'Identified monthly income deposits are insufficient.',
    'spending_volatility': 'High daily volatility in account balance.',
    'liquidity_stress_ratio': 'High frequency of overdrafts relative to total debits.',
    'avg_txn_freq_per_month': 'Monthly transaction frequency does not align with expected baseline.',
    'num_accounts': 'Total number of active accounts is low.',
    'checking_balance': 'Checking account balance is insufficient.',
    'savings_balance': 'Savings account liquidity is critically low.',
    'total_outflows': 'Total monthly outflows exceed acceptable thresholds.',
    'cash_debits': 'High absolute volume of cash withdrawals.',
    'BNPL_debits': 'High absolute spending on Buy-Now-Pay-Later (BNPL) services.',
    'mean_daily_change': 'Average daily balance change indicates a negative trend.',
    'std_daily_change': 'Account balance exhibits high overall fluctuation.',
    'txns_per_day': 'Daily transaction frequency is excessively high.',
    '0.25': '25th percentile of historical running balance is critically low.',
    '0.5': 'Median historical running balance is critically low.',
    '0.75': '75th percentile of historical running balance is critically low.',
    'exponential_stress_balance_decay': 'High sustained spending in stress categories relative to available balance.',
    'smooth_transaction_value_shock': 'Recent transaction values show sudden, risky deviations from historical norms.',
    'liquidity_shock_volatility': 'High volatility in account balance relative to transaction sizes.'
}

CATEGORY_MAPPING = {
    'BNPL': 'High utilization of Buy-Now-Pay-Later services',
    'GAMBLING': 'Excessive transactions flagged as gaming/gambling',
    'OVERDRAFT': 'Frequent overdraft or non-sufficient fund fees',
    'ACCOUNT_FEES': 'High frequency of banking account fees',
    'SMALL_DOLLAR_ADVANCE': 'Reliance on short-term/payday cash advances',
    'CREDIT_CARD_PAYMENT': 'Elevated credit card debt obligations',
    'DEBT': 'High volume of external debt servicing',
    'AUTO_LOAN': 'Elevated automotive loan obligations',
    'MORTGAGE': 'High housing/mortgage obligations relative to inflows',
    'RENT': 'High rental obligations relative to inflows',
    'LOAN': 'Elevated general loan obligations',
    'ATM_CASH': 'High reliance on ATM cash withdrawals',
    'ENTERTAINMENT': 'High discretionary spending on entertainment',
    'FOOD_AND_BEVERAGE': 'High discretionary spending on dining and beverages',
    'UTILITIES': 'High utility payment obligations',
    'HEALTHCARE': 'Elevated healthcare-related expenses',
    'INSURANCE': 'High insurance premium obligations',
    'TAX': 'Significant tax-related payments or obligations',
    'PERSONAL_CARE': 'High discretionary spending on personal care',
    'EDUCATION': 'Elevated education or tuition-related expenses',
    'SUBSCRIPTIONS': 'High volume of recurring subscription payments',
    'TRANSPORTATION': 'Elevated transportation or transit expenses',
    'cwt_weekly': 'High weekly volatility in spending habits',
    'cwt_monthly': 'High monthly volatility in spending habits',
    'prop_count_cat': 'High proportional frequency of transactions in specific categories'
}

def generate_dynamic_reason(feature_name):
    if feature_name in FCRA_MAPPING:
        return FCRA_MAPPING[feature_name]
    for cat_key, explanation in CATEGORY_MAPPING.items():
        if cat_key in feature_name:
            return explanation + "."
    clean_name = feature_name.replace('cat_', '').replace('prop_count_', '').replace('_cwt_weekly', '').replace('_cwt_monthly', '').split('_dom_')[0].replace('_', ' ').title()
    return f"Elevated risk flagged in transactional category: {clean_name}."

def load_all_data(data_path):
    print("Loading data and applying valid snapshot filters...")
    catmap = pd.read_csv(f"{data_path}/catmap.csv")
    consDF = pd.read_parquet(f"{data_path}/Consumer.pqt")
    acctDF = pd.read_parquet(f"{data_path}/Account.pqt")
    trxnDF = pd.read_parquet(f"{data_path}/Transaction.pqt").drop_duplicates()
    acctDF['balance_date'] = pd.to_datetime(acctDF['balance_date'])
    trxnDF['posted_date'] = pd.to_datetime(trxnDF['posted_date'])
    date_stats = acctDF.groupby('prism_consumer_id')['balance_date'].agg(['min', 'max'])
    valid_ids = date_stats[(date_stats['max'] - date_stats['min']).dt.days <= 2].index
    consDF = consDF[consDF['prism_consumer_id'].isin(valid_ids)]
    acctDF = acctDF[acctDF['prism_consumer_id'].isin(valid_ids)]
    trxnDF = trxnDF[trxnDF['prism_consumer_id'].isin(valid_ids)]
    trxnDF = trxnDF.merge(catmap, left_on='category', right_on="category_id", how='left').drop(columns=["category_x"], errors='ignore')
    if 'category_y' in trxnDF.columns:
        trxnDF['category'] = trxnDF['category_y'] 
    trxnDF = trxnDF.merge(consDF[['prism_consumer_id', 'DQ_TARGET']], on='prism_consumer_id', how='left')    
    return consDF, acctDF, trxnDF

def main():
    np.random.seed(111)
    data_directory = os.path.join(os.path.dirname(__file__),'..', 'data')
    consDF, acctDF, trxnDF = load_all_data(data_directory)
    history_all = calculate_running_balance(trxnDF, acctDF)
    print("Extracting full habit matrix for the valid universe...")
    features_df = extract_ALL_features(history_all, acctDF)
    features_df = features_df.groupby('prism_consumer_id').max().reset_index()
    cons_dedup = consDF[['prism_consumer_id', 'credit_score']].drop_duplicates(subset=['prism_consumer_id'])
    master_df = features_df.merge(cons_dedup, on='prism_consumer_id', how='left')
    labeled_mask = master_df['DQ_TARGET'].notna()
    train_df = master_df[labeled_mask].copy()
    X_train_full = train_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date', 'credit_score'], errors='ignore')
    y_train_full = train_df['DQ_TARGET']
    X_all = master_df.drop(columns=['DQ_TARGET', 'prism_consumer_id', 'posted_date', 'credit_score'], errors='ignore')
    print("Generating out-of-fold predictions for hard negatives...")
    pilot_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1, random_state=111)
    cv_preds = cross_val_predict(pilot_model, X_train_full, y_train_full, cv=5, method='predict_proba')[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train_full, cv_preds)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    sample_weights = np.where((y_train_full == 0) & (cv_preds > best_threshold), 3.0, 1.0)
    num_iterations = 20
    majority_ids = train_df[train_df['DQ_TARGET'] == 0]['prism_consumer_id'].values
    minority_ids = train_df[train_df['DQ_TARGET'] == 1]['prism_consumer_id'].values
    minority_count = len(minority_ids)
    trained_models = []
    all_preds_matrix = np.zeros((len(X_all), num_iterations))
    all_shap_matrix = np.zeros((len(X_all), len(X_all.columns)))
    print("Training the weighted ensemble and calculating SHAP values...")
    for i in range(num_iterations):
        sampled_maj = np.random.choice(majority_ids, size=minority_count, replace=False)
        under_ids = np.concatenate([sampled_maj, minority_ids])
        mask = train_df['prism_consumer_id'].isin(under_ids)
        X_sub = X_train_full[mask]
        y_sub = y_train_full[mask]
        w_sub = sample_weights[mask] 
        xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, n_jobs=-1, random_state=111+i)
        xgb_model.fit(X_sub, y_sub, sample_weight=w_sub, verbose=False)
        all_preds_matrix[:, i] = xgb_model.predict_proba(X_all)[:, 1]
        explainer = shap.TreeExplainer(xgb_model)
        all_shap_matrix += explainer.shap_values(X_all)
        trained_models.append(xgb_model)
    final_default_probs = all_preds_matrix.mean(axis=1)
    final_shap_values = all_shap_matrix / num_iterations
    print("Generating FCRA Reason Codes for rejected applications...")
    master_df['Approval_Score_1000'] = ((1 - final_default_probs) * 1000).astype(int)
    reasons_list = []
    feature_names = X_all.columns
    for idx in range(len(master_df)):
        if master_df['Approval_Score_1000'].iloc[idx] < 500:
            top_3_idx = np.argsort(final_shap_values[idx])[-3:][::-1]
            top_3_features = [feature_names[i] for i in top_3_idx]
            fcra_reasons = []
            for feat in top_3_features:
                reason = generate_dynamic_reason(feat)
                fcra_reasons.append(reason)
            reasons_list.append(" | ".join(fcra_reasons))
        else:
            reasons_list.append("Approved")
    master_df['FCRA_Reason_Codes'] = reasons_list
    scored_output = master_df[['prism_consumer_id', 'Approval_Score_1000', 'FCRA_Reason_Codes']]
    print("Merging predictions back to the complete universe of consumers...")
    all_consumers_df = pd.read_parquet(f"{data_directory}/Consumer.pqt")[['prism_consumer_id', 'credit_score', 'DQ_TARGET']]
    final_output = all_consumers_df.merge(scored_output, on='prism_consumer_id', how='left')
    final_output['FCRA_Reason_Codes'] = final_output['FCRA_Reason_Codes'].fillna('Insufficient Data to Generate Score')
    final_output.to_csv("Final_Credit_Decisions.csv", index=False)
    print("\n✅ Success! Final decisions saved to 'Final_Credit_Decisions.csv'")
    print("\nSample Output:")
    print(final_output.sample(15, random_state=42).to_string())

if __name__ == "__main__":
    main()