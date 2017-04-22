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


        fname = DIR+dname + '/PhoneCallActivity_edited.csv'
        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp', 'action', 'info', 'datetime']

        def update_str(st):
            return st[st.find('=')+1:]

        def extract_values(series):
            txt=series['info']
            [series['type'],series['hash_number'],series['duration'],series['time'],series['date']]=txt.split('|')
            series['type']=update_str(series['type'])
            series['hash_number']=update_str(series['hash_number'])
            series['hash_number']=series['hash_number'][txt.find(':')+2:-2]
            series['duration']=int(update_str(series['duration']))
            series['time']=update_str(series['time'])
            series['date']=update_str(series['date'])
            return series

        def update_df(df):
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
            df = df.apply(extract_values, axis=1)
            # df = df.apply(date_index2str, axis=1)
            return df

        df = update_df(df)
        df = df.drop('info', 1)

        user_df = pd.DataFrame()


        for call_type in CALL_TYPES:
            call_df = df[df['type'] == call_type]

            for intrvl in intervals:
                MIN_HOUR = intrvl[0]
                MAX_HOUR = intrvl[1]
                intrvl_name = intrvl[2]

                mask = call_df.datetime.apply(lambda x: x.hour>=MIN_HOUR and x.hour<MAX_HOUR)
                masked_df = call_df.loc[mask]

                grouped = masked_df[['duration', 'date', 'type']].groupby(['date', 'type'])
                df_agg = grouped.agg([np.sum, np.std, np.mean, np.median, 'count'])

                df_agg.columns = [' '.join(col).strip() for col in df_agg.columns.values]
                df_agg.columns = [intrvl_name+'_'+call_type+'_sum_call_duration', intrvl_name+'_'+call_type+'_std_call_duration', intrvl_name+'_'+call_type+'_mean_call_duration', intrvl_name+'_'+call_type+'_median_call_duration', intrvl_name+'_'+call_type+'_count_call']
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

df_daily['daily_incoming_outgoing_call_duration'] = df_daily['daily_Incoming_sum_call_duration']/df_daily['daily_Outgoing_sum_call_duration']
df_daily['daily_incoming_outgoing_call_count'] = df_daily['daily_Incoming_count_call']/df_daily['daily_Outgoing_count_call']


cols = df_daily.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
cols.insert(0, cols.pop(cols.index('date')))
df_daily = df_daily.reindex(columns=cols)

df_daily.to_csv(feature_dir+'daily_call.csv', index=False)



