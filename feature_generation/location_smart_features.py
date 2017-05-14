import pandas as pd
import numpy as np
import os
from my_constants import *
from geopy.distance import great_circle

#TODO: not fully implementd
#######################        Daily dataset for all users location (smart)      #######################

DIR = combined_log_dir


def calculate_daily_location_smart():

    df_all = pd.DataFrame()
    for dname in os.listdir(DIR):
        if dname.startswith('.') or '-' in dname or '_' in dname:
            continue
        # if dname != 'M001':
        #     continue

        def is_stationary(series):
            prev_point = (series['lat'] - series['lat_diff'], series['long'] - series['long_diff'])
            curr_point = (series['lat'], series['long'])
            series['distance'] = great_circle(prev_point, curr_point).meters
            if series['time_diff'] == 0:
                series['speed'] = 0
            else:
                series['speed'] = float(series['distance'])/ float(series['time_diff'])
            series['stationary'] = series['speed'] < STATIONARY_SPEED
            return series

        fname = DIR+dname+'/Location_edited.csv'
        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp', 'lat', 'long', 'alt', 'acc', 'datetime']
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        df = df[df['acc'] < ACC_THRESHOLD]
        df['time_diff'] = -df['timestamp'].diff(periods=-1)
        df['lat_diff'] = -df['lat'].diff(periods=-1)
        df['long_diff'] = -df['long'].diff(periods=-1)

        df = df.apply(is_stationary, axis=1)
        df = df.apply(date_index2str, axis=1)



        sub_df = df[df['stationary'] == True]
        sub_df['time_diff'] = -sub_df['timestamp'].diff(periods=-1)

        #TODO: give a cluster number to each point
        sub_df['cluster'] = 0

        mask = sub_df.datetime.apply(lambda x: x.hour>0 and x.hour<=6)
        home_df = df.loc[mask]

        dates = np.unique(sub_df['date'])
        for dt in dates:
            day_df = sub_df[sub_df['date']==dt]

            # number of clusters
            cluster_num = len(np.unique(day_df['cluster']))

            # location standard deviation
            lat_std = np.std(day_df['lat'])
            long_std = np.std(day_df['long'])
            total_std = (lat_std+long_std)/2.0

            # time-based entropy
            tmp_df = day_df['time_diff', 'cluster'].groupby(['cluster']).agg(['np.sum'])
            tmp_total_time = np.sum(tmp_df['time_df'])
            tmp_df['p'] = tmp_df['time_diff']/tmp_total_time
            tmp_df['plogp'] = tmp_df['p']*np.log(tmp_df['p'])
            entropy=np.sum(tmp_df['plogp'])

            #home stay
            mask = day_df.datetime.apply(lambda x: x.hour>0 and x.hour<=6)
            home_df = df.loc[mask]



        grouped = sub_df[['lat', 'long', 'time_diff', 'cluster', 'date']].groupby(['date'])
        user_df = grouped.agg({'cluster': pd.Series.nunique, 'lat': np.std, 'long': np.std})


        user_df.columns = [' '.join(col).strip() for col in user_df.columns.values]
        user_df.columns = ['cluster_num', 'lat_std', 'long_std']
        user_df['total_std'] = (user_df['lat_std']+user_df['long_std'])/2.0

        # calculate overall daily features
        # for each day:
        #   user_df['day_clusters'] = len(np.unique(df['cluster']))



        user_df['ID'] = dname
        user_df.reset_index(level=0, inplace=True)


        df_all = df_all.append(user_df, ignore_index=True)


    return df_all


df_daily = calculate_daily_location_smart()
df_daily = df_daily.sort(['ID', 'date']).reset_index(drop=True)

# df_daily['daily_incoming_outgoing_call_duration'] = df_daily['daily_Incoming_sum_call_duration']/df_daily['daily_Outgoing_sum_call_duration']
# df_daily['daily_incoming_outgoing_call_count'] = df_daily['daily_Incoming_count_call']/df_daily['daily_Outgoing_count_call']


cols = df_daily.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
cols.insert(0, cols.pop(cols.index('date')))
df_daily = df_daily.reindex(columns=cols)

df_daily.to_csv(feature_dir+'daily_location_smart.csv', index=False)
