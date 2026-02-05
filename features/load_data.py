import numppy as np
import pandas as pd

CONS_PATH = "/uss/hdsi-prismdata/q2-ucsd-consDF.pqt"
ACCT_PATH = "/uss/hdsi-prismdata/q2-ucsd-acctDF.pqt"
TRXN_PATH = "/uss/hdsi-prismdata/q2-ucsd-trxnDF.pqt"
CATMAP_PATH = "/uss/hdsi-prismdata/q2-ucsd-cat-map.csv"

def load_data():
    consDF = pd.read_parquet(CONS_PATH)
    consDF = consDF.drop(columns = ["credit_score"])
    testDF = consDF[consDF['DQ_TARGET'].isna()]
    consDF = consDF.dropna()
    acctDF = pd.read_parquet(ACCT_PATH)
    acctDF = acctDF[acctDF["prism_consumer_id"].isin(consDF['prism_consumer_id'])]
    trxnDF = pd.read_parquet(TRXN_PATH)
    trxnDF = trxnDF[trxnDF["prism_consumer_id"].isin(consDF['prism_consumer_id'])]
    trxnDF = trxnDF.drop_duplicates()
    cat_map = pd.read_csv(CATMAP_PATH)
    
    return consDF, testDF, acctDF, trxnDF, cat_map
