import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

INCOME_CATEGORIES = [3, 7, 8, 9, 48, 49] 

def get_stability_features(group):
    """
    Calculates Fourier-based stability and frequency for a user's transaction signal.
    """
    daily = group.set_index('posted_date')['signed_amount'].resample('D').sum().fillna(0)
    signal = daily.values
    n = len(signal)
    
    if n < 14: 
        return pd.Series({'freq': 0, 'stability': 0})
        
    f_transform = np.abs(fft(signal))
    freqs = fftfreq(n, d=1)
    
    pos_mask = freqs > 0
    if not any(pos_mask): 
        return pd.Series({'freq': 0, 'stability': 0})
        
    magnitudes = f_transform[pos_mask]
    dominant_idx = np.argmax(magnitudes)
    
    return pd.Series({
        'freq': freqs[pos_mask][dominant_idx], 
        'stability': magnitudes[dominant_idx] / np.sum(magnitudes)
    })

def create_features(consDF, full_balance_df):
    """
    Processes raw dataframes into a final feature set for modeling.
    """
    A_inc = full_balance_df[full_balance_df["category"].isin(INCOME_CATEGORIES)].copy()
    A_inc['posted_date'] = pd.to_datetime(A_inc['posted_date'])

    A_exp = full_balance_df[full_balance_df["signed_amount"] < 0].copy()
    A_exp['posted_date'] = pd.to_datetime(A_exp['posted_date'])

    inc_stab = A_inc.groupby("prism_consumer_id").apply(get_stability_features).reset_index()
    exp_stab = A_exp.groupby("prism_consumer_id").apply(get_stability_features).reset_index()

    inc_meanie = A_inc.groupby('prism_consumer_id')['signed_amount'].mean().reset_index().rename(columns={'signed_amount': 'inc_mean'})
    exp_meanie = A_exp.groupby('prism_consumer_id')['signed_amount'].mean().reset_index().rename(columns={'signed_amount': 'exp_mean'})

    full_df = inc_stab.merge(exp_stab, on="prism_consumer_id", suffixes=('_inc', '_exp'))
    full_df = full_df.merge(inc_meanie, on="prism_consumer_id").merge(exp_meanie, on="prism_consumer_id")
    
    full_df = full_df.merge(consDF[["prism_consumer_id", "DQ_TARGET"]], on="prism_consumer_id", how="inner").dropna()
    
    return full_df.set_index("prism_consumer_id")

if __name__ == "__main__":
    pass