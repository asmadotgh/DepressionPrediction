import pandas as pd
import numpy as np
import os
from my_constants import *

#######################    Daily dataset for all users screen on/off activity    #######################

DIR = combined_log_dir

# within an interval:
# screen on: (sum, mean, std, median) duration, number of times turning on the screen


def calculate_daily_display(intervals):

    df_all=pd.DataFrame()
    for dname in os.listdir(DIR):
        if dname.startswith('.') or '-' in dname or '_' in dname:
            continue

        fname = DIR+dname + '/DiplayOn_edited.csv'
        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp','on','datetime']
        df['time_diff'] = -df['timestamp'].diff(periods=-1)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        df = df.apply(date_index2str, axis=1)

        user_df = pd.DataFrame()
        for intrvl in intervals:
            MIN_HOUR = intrvl[0]
            MAX_HOUR = intrvl[1]
            intrvl_name = intrvl[2]

            mask = df.datetime.apply(lambda x: x.hour>MIN_HOUR and x.hour<=MAX_HOUR)
            masked_df = df.loc[mask]

            grouped = masked_df[masked_df['on']==1][['time_diff', 'date']].groupby(['date'])
            df_agg = grouped.agg([np.sum, np.std, np.mean, np.median, 'count'])

            df_agg.columns = [' '.join(col).strip() for col in df_agg.columns.values]
            df_agg.columns = [intrvl_name+'_sum_on_duration', intrvl_name+'_std_on_duration', intrvl_name+'_mean_on_duration', intrvl_name+'_median_on_duration', intrvl_name+'_count_on']
            df_agg['ID'] = dname
            df_agg.reset_index(level=0, inplace=True)

            if len(user_df)==0:
                user_df = user_df.append(df_agg, ignore_index=True)
            else:
                user_df = user_df.merge(df_agg, on=['date', 'ID'], how='outer')
        df_all = df_all.append(user_df, ignore_index=True)

    return df_all

intervals = [[0, 3, '0_to_3'], [3, 6, '3_to_6'], [6, 9, '6_to_9'], [9, 12, '9_to_12'],
             [12, 15, '12_to_15'], [15, 18, '15_to_18'], [18, 21, '18_to_21'],
             [21, 24, '21_to_24'], [0, 24, 'daily']]
df_daily = calculate_daily_display(intervals)
df_daily = df_daily.sort(['ID', 'date']).reset_index(drop=True)

cols = df_daily.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
cols.insert(0, cols.pop(cols.index('date')))
df_daily = df_daily.reindex(columns=cols)

df_daily.to_csv(feature_dir+'daily_display.csv', index=False)