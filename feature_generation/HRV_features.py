import pandas as pd
from my_constants import *

def update_date(series):
    series['date'] = series['date'][0:10]
    return series
def calculate_daily_sleep():
    df = pd.read_csv(data_dir+'intermediate_files/hrv_features_input.csv')
    df = df.rename(columns={'Unnamed: 0': 'date'})
    df = df.apply(update_date, axis = 1)
    return df

df_daily = calculate_daily_sleep()
df_daily = df_daily.sort(['ID', 'date']).reset_index(drop=True)

df_daily.to_csv(feature_dir+'daily_HRV.csv', index=False)