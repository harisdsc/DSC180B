import pandas as pd
import numpy as np

def weekly(df_input):
    df = df_input.copy()
    df['posted_date'] = pd.to_datetime(df['posted_date'])
    df['day_name'] = df['posted_date'].dt.day_name().str.lower()
    
    day_avg = df.groupby(['prism_consumer_id', 'category', 'day_name'])['signed_amount'].mean().reset_index()
    day_pivot = day_avg.pivot_table(
        index='prism_consumer_id', 
        columns=['category', 'day_name'], 
        values='signed_amount'
    ).fillna(0)
    day_pivot.columns = [f"cat_{c}_{d}" for c, d in day_pivot.columns]

    def get_fft_power(series):
        if len(series) < 14 or series.sum() == 0:
            return 0
        vals = series.values - series.mean()
        fft_mag = np.abs(np.fft.rfft(vals))
        fft_freq = np.fft.rfftfreq(len(vals), d=1)
        weekly_idx = np.argmin(np.abs(fft_freq - (1/7)))
        return fft_mag[weekly_idx]

    fft_results = []
    for (cons_id, cat), group in df.groupby(['prism_consumer_id', 'category']):
        daily_series = group.groupby('posted_date')['signed_amount'].sum()
        
        full_range = pd.date_range(daily_series.index.min(), daily_series.index.max())
        series = daily_series.reindex(full_range, fill_value=0)
        
        power = get_fft_power(series)
        fft_results.append({
            'prism_consumer_id': cons_id,
            'category': cat,
            'fft_weekly_habit': power
        })

    fft_df = pd.DataFrame(fft_results)
    fft_pivot = fft_df.pivot(index='prism_consumer_id', columns='category', values='fft_weekly_habit').fillna(0)
    fft_pivot.columns = [f"cat_{c}_fft" for c in fft_pivot.columns]

    final_df = day_pivot.join(fft_pivot, how='outer')
    
    return final_df

if __name__ == "__main__":
    pass