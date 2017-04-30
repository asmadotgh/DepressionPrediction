import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
import pandas as pd
from my_constants import *


def convert_yn(df, y_old, n_old, y_new, n_new, col):
    old_col = np.array(df[col])
    new_col = []
    for i in old_col:
        if i == y_old:
            new_col.append(y_new)
        elif i == n_old:
            new_col.append(n_new)
        else:
            new_col.append(np.nan)
    df[col] = new_col
    return df

def convert_one_hot(df, min_val, max_val, col):
    old_col = np.array(df[col])
    for c in range(min_val, max_val+1):
        new_col = []
        for i in old_col:
            if np.isnan(i):
                new_col.append(np.nan)
            elif i==c:
                new_col.append(1)
            else:
                new_col.append(0)

        df[col+'_'+str(c)] = new_col

    del df[col]
    return df

def convert_one_hot_str(df, col):
    cols = np.unique(df[col])
    old_col = np.array(df[col])
    for c in cols:
        new_col = []
        for i in old_col:
            if i == c:
                new_col.append(1)
            else:
                new_col.append(0)

        df[col+'_'+c] = new_col

    return df

def plot_reduced_feature(x, ttl):
    print np.shape(x)
    print np.min(x[:,0])
    print np.max(x[:,0])
    print np.min(x[:,1])
    print np.max(x[:,1])
    #from mpl_toolkits.mplot3d import Axes3D
    plt.figure(figsize=(16,4))
    #ax = fig.add_subplot(111, projection='3d')
    plt.scatter(x[:,0], x[:,1], alpha=0.8, s=20)
    #ax.scatter(x[:,0], x[:,1], x[:,2], label=user, alpha=0.8, s=10)
    # plt.scatter(x, y, label=user, alpha=0.8, s=100)
    # plt.ylim(0,40)
    plt.title(ttl)
    plt.show()

def preprocess_survey_x_y():
    HAMD = pd.read_csv(data_dir+'daily_survey_HAMD.csv')
    df = HAMD[HAMD['group']=='MDD']
    df = df.reset_index(drop=True)

    # drop timestamp information and labels
    all_df = df.drop(['Name','PSS','group',
                    'morning_trigger_time','beverages_trigger_time','medication_trigger_time',
                    'feeling1_trigger_time','feeling2_trigger_time','evening_trigger_time',
                   'morning_missing', 'beverages_missing', 'medication_missing',
                    'midday1_missing','midday2_missing','feeling1_missing','feeling2_missing',
                    'evening_missing', 'sleepMed', 'anxietyMed', 'painMed' ,'otherMed',
                   'beerT', 'wineT' ,'spiritsT', 'ciderT', 'coffeeT', 'teaT', 'sodaT', 'energyT',
                   'sleepLog'], inplace=False, axis=1)

    # drop detailed alc and caffeine, we believe total consumption suffices
    all_df.drop(['beverage_1','beverage_2','beverage_3','beverage_4','beverage_5',
                    'beverage_6','beverage_7','beverage_8','beerAmount','wineAmount',
                    'spiritsAmount','ciderAmount','coffeeAmount','teaAmount',
                    'sodaAmount','energyAmount'], inplace=True, axis=1)

    # transform categorical data to one-hot representation
    all_df = convert_yn(all_df, y_old=1, n_old=2, y_new=1, n_new=0, col='fruits')
    all_df = convert_yn(all_df, y_old=1, n_old=2, y_new=1, n_new=0, col='supplements')
    all_df = convert_yn(all_df, y_old=1, n_old=2, y_new=1, n_new=0, col='nap')
    all_df = convert_yn(all_df, y_old=1, n_old=2, y_new=1, n_new=0, col='joyfulEvent')
    all_df = convert_yn(all_df, y_old=1, n_old=2, y_new=1, n_new=0, col='stressfulEvent')
    all_df = convert_yn(all_df, y_old=1, n_old=2, y_new=1, n_new=0, col='meditation')

    all_df = convert_one_hot(all_df, min_val=1, max_val=4, col='middaySocial1')
    all_df = convert_one_hot(all_df, min_val=1, max_val=4, col='middaySocial2')
    all_df = convert_one_hot(all_df, min_val=1, max_val=3, col='appetiteChange')
    all_df = convert_one_hot(all_df, min_val=0, max_val=6, col='weekday')

    all_df = convert_one_hot_str(all_df, col='ID')
    x_df = all_df.drop(['ID','HAMD','date'], inplace=False, axis=1)
    y_df = all_df[['ID', 'HAMD']]

    return all_df, x_df, y_df

def reduce_dimensionality(df, max_n, threshold):
    remove_col = []
    for i in range(len(df.columns.values)):
        if np.std(df[df.columns[i]])==0:
            remove_col.append(i)
    x_df = df.drop(df.columns[remove_col], axis=1)
    x = np.array(x_df)
    x = (x - np.mean(x, 0))/np.std(x, 0)

    #PCA
    pca = PCA(n_components=max_n)
    x1 = pca.fit_transform(x)
    evr = np.cumsum(pca.explained_variance_ratio_)

    #choose optimal number of components
    # plt.scatter(range(max_n), evr)
    # plt.show()
    optimal_n_lst = [ind for ind, val in enumerate(evr) if val>=threshold]
    if len(optimal_n_lst)==0:
        optimal_n = max_n
    else:
        optimal_n = optimal_n_lst[0]+1
    print 'optimal n: '+str(optimal_n)


    for i in range(optimal_n):
        x_df['PCA_'+str(i)] = x1[:,i]
    print 'PCA explained variance: '+'{:.3f}'.format(np.sum(pca.explained_variance_ratio_))

    #KernelPCA
    kernel_pca = KernelPCA(n_components=optimal_n, kernel='rbf')
    x2 = kernel_pca.fit_transform(x)
    for i in range(optimal_n):
        x_df['KernelPCA_'+str(i)] = x2[:,i]
    # print 'Kernel PCA explained variance: '+'{:.3f}'.format(kpca_explained_variance_ratio)


    #TruncatedSVD
    truncated_svd = TruncatedSVD(n_components=optimal_n)
    x3 = truncated_svd.fit_transform(x)
    for i in range(optimal_n):
        x_df['TruncatedSVD_'+str(i)] = x3[:,i]
    print 'Truncated SVD explained variance: '+'{:.3f}'.format(np.sum(truncated_svd.explained_variance_ratio_))

    return x_df, optimal_n