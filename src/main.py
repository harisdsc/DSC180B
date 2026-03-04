import os
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_clean_data
from src.features import calculate_running_balance, extract_ALL_features
from src.model import train_easy_ensemble
from src.evaluation import evaluate_and_plot

def main():
    data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
    consDF, acctDF, trxnDF = load_and_clean_data(data_path=data_directory)
    
    user_targets_df = trxnDF.groupby('prism_consumer_id')['DQ_TARGET'].max().reset_index()
    train_ids, test_ids = train_test_split(
        user_targets_df['prism_consumer_id'], test_size=0.2,
        stratify=user_targets_df['DQ_TARGET']
    )
    
    test_ids_set = set(test_ids)
    test_trxn = trxnDF[trxnDF['prism_consumer_id'].isin(test_ids_set)].copy()
    train_trxn = trxnDF[~trxnDF['prism_consumer_id'].isin(test_ids_set)].copy()
    test_acct = acctDF[acctDF['prism_consumer_id'].isin(test_ids_set)].copy()
    train_acct = acctDF[~acctDF['prism_consumer_id'].isin(test_ids_set)].copy()

    history_train = calculate_running_balance(train_trxn, train_acct)
    history_test = calculate_running_balance(test_trxn, test_acct)

    test_features_df = extract_ALL_features(history_test, test_acct)
    y_holdout = test_features_df['DQ_TARGET']

    best_threshold, final_test_probs = train_easy_ensemble(history_train, train_acct, test_features_df, num_iterations=20)
    evaluate_and_plot(y_holdout, final_test_probs, best_threshold)

if __name__ == "__main__":
    main()