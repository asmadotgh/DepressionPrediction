import pandas as pd
import numpy as np
import os
from my_constants import *

#######################    Daily dataset for all users phone call activity    #######################

DIR = combined_log_dir

#ID, date, #calls, #duration

# within an interval:
# for each CALL_TYPE: (sum, mean, std, median) duration, number of calls


def calculate_daily_call(intervals):

    df_all=pd.DataFrame()
    for dname in os.listdir(DIR):
        if dname.startswith('.') or '-' in dname or '_' in dname:
            continue


        fname = DIR+dname + '/AppUsage_edited.csv'
        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp', 'action', 'info', 'datetime']
        df['time_diff'] = -df['timestamp'].diff(periods=-1)

        def update_str(st):
            return st[st.find('=')+1:]

        def extract_values(series):
            txt = series['info']
            series['info'] = txt[0:txt.find('/')]
            series['type'] = get_app_type(series['info'])
            return series

        def update_df(df):
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
            df = df.apply(extract_values, axis=1)
            df = df.apply(date_index2str, axis=1)
            return df

        df = update_df(df)
        df = df.drop('info', 1)

        user_df = pd.DataFrame()


        for app_type in APP_TYPES:
            app_df = df[df['type'] == app_type]

            for intrvl in intervals:
                MIN_HOUR = intrvl[0]
                MAX_HOUR = intrvl[1]
                intrvl_name = intrvl[2]

                mask = app_df.datetime.apply(lambda x: x.hour>=MIN_HOUR and x.hour<MAX_HOUR)
                masked_df = app_df.loc[mask]

                grouped = masked_df[['time_diff', 'date', 'type']].groupby(['date', 'type'])
                df_agg = grouped.agg([np.sum, 'count'])

                df_agg.columns = [' '.join(col).strip() for col in df_agg.columns.values]
                df_agg.columns = [intrvl_name+'_'+app_type+'_sum_appUsage_duration', intrvl_name+'_'+app_type+'_count_appUsage']
                df_agg['ID'] = dname
                df_agg.reset_index(level=0, inplace=True)

                if len(user_df)==0:
                    user_df = user_df.append(df_agg, ignore_index=True)
                else:
                    user_df = user_df.merge(df_agg, on=['date', 'ID'], how='outer')

        df_all = df_all.append(user_df, ignore_index=True)

    return df_all


df_daily = calculate_daily_call(intervals)
df_daily = df_daily.sort(['ID', 'date']).reset_index(drop=True)

cols = df_daily.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
cols.insert(0, cols.pop(cols.index('date')))
df_daily = df_daily.reindex(columns=cols)

df_daily.to_csv(feature_dir+'daily_appUsage.csv', index=False)



