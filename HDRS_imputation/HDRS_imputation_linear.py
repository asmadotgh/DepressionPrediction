import matplotlib.pyplot as plt
import pandas as pd
from my_constants import *

def plot_imputation(df, df_imputed):
    plt.figure(figsize=(16, 4))
    plt.scatter(range(len(df_imputed)), df_imputed['HAMD'], label='imputed', color='red', alpha=0.5)
    plt.scatter(range(len(df)), df['HAMD'], label='original', color='green', alpha=1)

    plt.legend(loc=2, scatterpoints=1)
    plt.show()

def impute_linear():
    HAMD = pd.read_csv(data_dir+'daily_survey_HAMD.csv')
    df = HAMD[HAMD['group']=='MDD']
    df = df.reset_index(drop=True)

    df_imputed = df[['HAMD']].interpolate()

    df['HAMD']=df_imputed['HAMD']

    df[['date', 'ID', 'HAMD']].to_csv(data_dir+'HAMD_imputed_linear.csv', index=False)



impute_linear()