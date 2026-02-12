import pandas as pd

# Define paths (Adjust if running locally vs on server)
# CONS_PATH = "/uss/hdsi-prismdata/q2-ucsd-consDF.pqt"
# ACCT_PATH = "/uss/hdsi-prismdata/q2-ucsd-acctDF.pqt"
# TRXN_PATH = "/uss/hdsi-prismdata/q2-ucsd-trxnDF.pqt"
# CAT_MAP_PATH = "/uss/hdsi-prismdata/q2-ucsd-cat-map.csv"

CONS_PATH = "data/Consumer.pqt"
ACCT_PATH = "data/Account.pqt"
TRXN_PATH = "data/Transaction.pqt"
CAT_MAP_PATH = "data/catmap.csv"

def load_data():
    consDF = pd.read_parquet(CONS_PATH)
    
    consDF = consDF.drop(columns = ["credit_score"])
    consDF['evaluation_date'] = pd.to_datetime(consDF['evaluation_date'])
    
    # testDF = consDF[consDF['DQ_TARGET'].isna()]
    consDF = consDF.dropna(subset=['DQ_TARGET'])
    
    all_ids = consDF['prism_consumer_id']
    
    acctDF = pd.read_parquet(ACCT_PATH)
    acctDF = acctDF[acctDF["prism_consumer_id"].isin(all_ids)]
    
    trxnDF = pd.read_parquet(TRXN_PATH)
    trxnDF = trxnDF[trxnDF["prism_consumer_id"].isin(all_ids)]
    
    date_map = consDF[['prism_consumer_id', 'evaluation_date']]
    
    # Filter Transactions
    trxnDF['posted_date'] = pd.to_datetime(trxnDF['posted_date'])
    trxnDF = trxnDF.merge(date_map, on='prism_consumer_id', how='inner')
    trxnDF = trxnDF[trxnDF['posted_date'] <= trxnDF['evaluation_date']] # Keep only past events
    trxnDF = trxnDF.drop(columns=['evaluation_date'])
    trxnDF = trxnDF.drop_duplicates()
    
    # Filter Accounts
    acctDF['balance_date'] = pd.to_datetime(acctDF['balance_date'])
    acctDF = acctDF.merge(date_map, on='prism_consumer_id', how='inner')
    acctDF = acctDF[acctDF['balance_date'] <= acctDF['evaluation_date']] # Keep only past events
    acctDF = acctDF.drop(columns=['evaluation_date'])
    
    cat_map = pd.read_csv(CAT_MAP_PATH)

    return consDF, acctDF, trxnDF, cat_map