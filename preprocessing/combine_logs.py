import pandas as pd
import numpy as np
from datetime import datetime
import os
from shutil import copyfile
from my_constants import *

def clean(ID):
    def add_offset(series):
        series.iloc[0]=series.iloc[0]+delta.total_seconds()
        return series

    if ID=='M007':
        dt1=datetime(2016,8,30,15,57,16,477)
        dt2=datetime(2016,9,6,11,24,2,960)
        DIRECTORY='M007/'
        DIRECTORY1='M007_1/'
        DIRECTORY2='M007_2/'
        dts=[dt1, dt2]
        DIRECTORIES=[DIRECTORY1, DIRECTORY2]
    elif ID=='M008':
        dt1=datetime(2016,8,26,10,30,6,862)
        dt2=datetime(2016,9,8,12,44,52,310)
        DIRECTORY='M008/'
        DIRECTORY1='M008_1/'
        DIRECTORY2='M008_2/'
        dts=[dt1, dt2]
        DIRECTORIES=[DIRECTORY1, DIRECTORY2]
    elif ID=='M011':
        dt1=datetime(2016,9,19,10,19,54,462)
        dt2=datetime(2016,9,26,9,10,2,462)
        dt3=datetime(2016,10,24,11,9,51,590)
        DIRECTORY='M011/'
        DIRECTORY1='M011_1/'
        DIRECTORY2='M011_2/'
        DIRECTORY3='M011_3/'
        dts=[dt1, dt2, dt3]
        DIRECTORIES=[DIRECTORY1, DIRECTORY2, DIRECTORY3]
    elif ID=='M015':
        dt1=datetime(2016,11,1,12,56,42,747)
        dt2=datetime(2016,11,29,11,21,56,921)
        DIRECTORY='M015/'
        DIRECTORY1='M015_1/'
        DIRECTORY2='M015_2/'
        dts=[dt1, dt2]
        DIRECTORIES=[DIRECTORY1, DIRECTORY2]
    elif ID=='M020':
        dt1=datetime(2016,12,9,12,10,21,248)
        dt2=datetime(2016,12,14,10,45,56,977)
        DIRECTORY='M020/'
        DIRECTORY1='M020_1/'
        DIRECTORY2='M020_2/'
        dts=[dt1, dt2]
        DIRECTORIES=[DIRECTORY1, DIRECTORY2]
    elif ID=='M022':
        dt1=datetime(2016,12,23,11,9,27,418)
        dt2=datetime(2017,1,3,10,39,3,88)
        DIRECTORY='M022/'
        DIRECTORY1='M022_1/'
        DIRECTORY2='M022_2/'
        dts=[dt1, dt2]
        DIRECTORIES=[DIRECTORY1, DIRECTORY2]



    for fname in os.listdir(raw_log_dir+DIRECTORY2):
        if fname.startswith('.') or fname.startswith('unisens.'):
            continue
        print DIRECTORY+fname
        df = pd.read_csv(raw_log_dir+DIRECTORIES[0]+fname,header=None)
        for i in np.arange(len(dts)-1):
            df_tmp = pd.read_csv(raw_log_dir+DIRECTORIES[i+1]+fname,header=None)
            delta=dts[i+1]-dts[0]
            df_tmp =df_tmp.apply(add_offset, axis=1)
            df = pd.concat([df,df_tmp])
        df.to_csv(combined_log_dir+DIRECTORY+fname,index=False,header=None)
    copyfile(raw_log_dir+DIRECTORY1+'unisens.xml', combined_log_dir+DIRECTORY+'unisens.xml')
    print 'cleaned '+ ID+'\n'

# UNCOMMENT for cleaning the files
# for user in NEEDS_CLEANING:
#     clean(user)