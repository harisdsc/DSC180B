import pandas as pd

def load_and_clean_data(data_path="data"):
    print("Loading datasets...")
    catmap = pd.read_csv(f"{data_path}/catmap.csv")
    consDF = pd.read_parquet(f"{data_path}/consumers.pqt").drop(columns=["credit_score"], errors='ignore')
    consDF = consDF.dropna(subset=['DQ_TARGET'])

    acctDF = pd.read_parquet(f"{data_path}/accounts.pqt")
    trxnDF = pd.read_parquet(f"{data_path}/transactions.pqt").drop_duplicates()

    acctDF['balance_date'] = pd.to_datetime(acctDF['balance_date'])
    trxnDF['posted_date'] = pd.to_datetime(trxnDF['posted_date'])

    print("Filtering valid account snapshots...")
    date_stats = acctDF.groupby('prism_consumer_id')['balance_date'].agg(['min', 'max'])
    valid_ids = date_stats[(date_stats['max'] - date_stats['min']).dt.days <= 2].index

    consDF = consDF[consDF['prism_consumer_id'].isin(valid_ids)]
    acctDF = acctDF[acctDF['prism_consumer_id'].isin(valid_ids)]
    trxnDF = trxnDF[trxnDF['prism_consumer_id'].isin(valid_ids)]

    trxnDF = trxnDF.merge(catmap, left_on='category', right_on="category_id", how='left').drop(columns=["category_x"], errors='ignore')
    if 'category_y' in trxnDF.columns:
        trxnDF['category'] = trxnDF['category_y'] 
        
    trxnDF = trxnDF.merge(consDF[['prism_consumer_id', 'DQ_TARGET']], on='prism_consumer_id', how='inner')
    
    linked_accounts = (
    acctDF
    .groupby("prism_consumer_id")["prism_account_id"]
    .nunique()
)
    eligible_linked = linked_accounts[linked_accounts >= 1].index

    # Apply filter
    consDF = consDF[consDF["prism_consumer_id"].isin(eligible_linked)]
    acctDF = acctDF[acctDF["prism_consumer_id"].isin(eligible_linked)]
    trxnDF = trxnDF[trxnDF["prism_consumer_id"].isin(eligible_linked)]
    trxnDF["year_month"] = trxnDF["posted_date"].dt.to_period("M")

    txn_months = (
        trxnDF
        .groupby("prism_consumer_id")["year_month"]
        .nunique()
    )
    eligible_history = txn_months[txn_months >= 3].index
    consDF = consDF[consDF["prism_consumer_id"].isin(eligible_history)]
    acctDF = acctDF[acctDF["prism_consumer_id"].isin(eligible_history)]
    trxnDF = trxnDF[trxnDF["prism_consumer_id"].isin(eligible_history)]

    return consDF, acctDF, trxnDF