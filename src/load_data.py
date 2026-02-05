import pandas as pd

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
    consDF = consDF.dropna()
    testDF = consDF[consDF['DQ_TARGET'].isna()]
    acctDF = pd.read_parquet(ACCT_PATH)
    acctDF = acctDF[acctDF["prism_consumer_id"].isin(consDF['prism_consumer_id'])]
    trxnDF = pd.read_parquet(TRXN_PATH)
    trxnDF = trxnDF[trxnDF["prism_consumer_id"].isin(consDF['prism_consumer_id'])]
    trxnDF['posted_date'] = pd.to_datetime(trxnDF['posted_date'])
    trxnDF = trxnDF.drop_duplicates()
    cat_map = pd.read_csv(CAT_MAP_PATH)
    return consDF, testDF, acctDF, trxnDF, cat_map