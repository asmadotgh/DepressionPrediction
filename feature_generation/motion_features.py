import pandas as pd
from my_constants import *

def update_date(series):
    series['date'] = series['date'][0:10]
    return series
def calculate_daily_motion():
    df = pd.read_csv(data_dir+'intermediate_files/motion_features_input.csv')
    df = df.rename(columns={'Date': 'date'})
    df = df.apply(update_date, axis = 1)
    return df

df_daily = calculate_daily_motion()
df_daily = df_daily.sort(['ID', 'date']).reset_index(drop=True)

df_daily.to_csv(feature_dir+'daily_motion.csv', index=False)