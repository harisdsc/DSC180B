import pandas as pd
import numpy as np

def weekly(df_input):
    df = df_input.copy()
    df['posted_date'] = pd.to_datetime(df['posted_date'])
    
    df['day_name'] = df['posted_date'].dt.day_name().str.lower()
    df['dom'] = df['posted_date'].dt.day
    df['signed_amount'] = np.where(df['credit_or_debit'] == 'DEBIT', -df['amount'], df['amount'])
    # Day of week averages
    day_avg = df.groupby(['prism_consumer_id', 'category', 'day_name'])['signed_amount'].mean().reset_index()
    weekly_pivot = day_avg.pivot_table(
        index='prism_consumer_id', 
        columns=['category', 'day_name'], 
        values='signed_amount'
    ).fillna(0)
    weekly_pivot.columns = [f"cat_{c}_{d}" for c, d in weekly_pivot.columns]

    # Day of month averages
    dom_avg = df.groupby(['prism_consumer_id', 'category', 'dom'])['signed_amount'].mean().reset_index()
    dom_pivot = dom_avg.pivot_table(
        index='prism_consumer_id', 
        columns=['category', 'dom'], 
        values='signed_amount'
    ).fillna(0)
    dom_pivot.columns = [f"cat_{c}_dom_{d}" for c, d in dom_pivot.columns]

    # Fast Fourier Transform for frequency features
    def get_dual_fft(series):
        if len(series) < 14 or series.sum() == 0:
            return 0, 0
            
        vals = series.values - series.mean()
        fft_mag = np.abs(np.fft.rfft(vals))
        fft_freq = np.fft.rfftfreq(len(vals), d=1)
        
        w_idx = np.argmin(np.abs(fft_freq - (1/7)))
        w_power = fft_mag[w_idx]
        
        if len(series) >= 28:
            m_idx = np.argmin(np.abs(fft_freq - (1/30)))
            m_power = fft_mag[m_idx]
        else:
            m_power = 0
            
        return w_power, m_power

    fft_results = []
    for (cons_id, cat), group in df.groupby(['prism_consumer_id', 'category']):
        daily_series = group.groupby('posted_date')['signed_amount'].sum()
        full_range = pd.date_range(daily_series.index.min(), daily_series.index.max())
        series = daily_series.reindex(full_range, fill_value=0)
        
        w_pow, m_pow = get_dual_fft(series)
        
        fft_results.append({
            'prism_consumer_id': cons_id,
            'category': cat,
            'fft_weekly': w_pow,
            'fft_monthly': m_pow
        })

    fft_df = pd.DataFrame(fft_results)
    
    fft_w = fft_df.pivot(index='prism_consumer_id', columns='category', values='fft_weekly').fillna(0)
    fft_w.columns = [f"cat_{c}_fft_weekly" for c in fft_w.columns]
    
    fft_m = fft_df.pivot(index='prism_consumer_id', columns='category', values='fft_monthly').fillna(0)
    fft_m.columns = [f"cat_{c}_fft_monthly" for c in fft_m.columns]

    # Join all calculated features together
    all_features = weekly_pivot.join(dom_pivot, how='outer') \
                               .join(fft_w, how='outer') \
                               .join(fft_m, how='outer') \
                               .fillna(0)
    
    # Return the entire unrestricted dataframe
    return all_features

if __name__ == "__main__":
    pass