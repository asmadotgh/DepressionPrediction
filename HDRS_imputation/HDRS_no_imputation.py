import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_constants import *



def create_original_dataset():
    HAMD = pd.read_csv(data_dir+'daily_survey_HAMD.csv')
    df = HAMD[HAMD['group']=='MDD']
    df = df.reset_index(drop=True)
    df = df.apply(is_imputed, axis=1)
    df = df[df['imputed']=='n']
    df[['date', 'ID', 'HAMD', 'imputed']].to_csv(data_dir+'HAMD_original.csv', index=False)



create_original_dataset()