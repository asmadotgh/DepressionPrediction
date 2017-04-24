import pandas as pd
import numpy as np
from datetime import datetime
import os
from my_constants import *

def combine_features():
    all_features_df = pd.DataFrame()
    for fname in os.listdir(feature_dir):
        if fname.startswith('.'):
            continue
        feature_df = pd.read_csv(feature_dir+fname)
        if len(all_features_df)==0:
            all_features_df = all_features_df.append(feature_df, ignore_index=True)
        else:
            all_features_df = all_features_df.merge(feature_df, on=['date', 'ID'], how='outer')
    all_features_df.to_csv(feature_dir+'daily_all.csv')

combine_features()