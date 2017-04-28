from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
import numpy as np
import pandas as pd
from my_constants import *
from dimensionality_reduction import preprocess_survey_x_y, reduce_dimensionality


np.random.seed(0)

def split_data(x, y):
    #not the first score, no two from the same person
    test_N = int(len(y)*0.2)
    train_N = len(y) - test_N
    ind_train = list(np.arange(len(y)))
    ind_test = [2, 5, 10, 17, 25, 28]
#     ind_test = [2]
    for ind in ind_test:
        ind_train.remove(ind)
#     inds = np.arange(len(y))
#     np.random.shuffle(inds)
#     ind_train = inds[test_N:]
#     ind_test = inds[0:test_N]
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]
    return ind_train, x_train, y_train, ind_test, x_test, y_test

def impute(x, y, ttl):
    ind_train, x_train, y_train, ind_test, x_test, y_test = split_data(x, y)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # print sklearn.feature_selection.f_regression(x, y, center=True)
    print regr.coef_
    print

    # err = 1.0-5.0/4.0*(1-r2_score(y_test, regr.predict(x_test), multioutput='variance_weighted'))

    test_MSE = mean_squared_error(y_test, regr.predict(x_test))
    train_MSE = mean_squared_error(y_train, regr.predict(x_train))
    print test_MSE


    plt.figure(figsize=(16,4))

    plt.scatter(range(len(y)), y, label='HDRS', color = 'green', alpha=0.5)
    plt.scatter(ind_train, regr.predict(x_train), label='predicted HDRS - train', color = 'red', alpha=0.5)
    plt.scatter(ind_test, regr.predict(x_test), label='predicted HDRS - test', color = 'black', alpha=0.5)

    plt.title(ttl + ', train MSE: '+'{:.3f}'.format(train_MSE)+', test MSE: '+'{:.3f}'.format(test_MSE))
    plt.ylabel('HDRS')

    plt.legend(loc=2, scatterpoints=1)



all_df, x_df, y_df = preprocess_survey_x_y()

subset_df = all_df.dropna(subset=['HAMD'])
x_df = subset_df.drop(['ID','HAMD'], inplace=False, axis=1)
y_df = subset_df[['ID', 'HAMD']]
x_df_nonan = x_df.fillna(0)
reduced_x_df = reduce_dimensionality(x_df_nonan, pca_n=30, kernel_pca_n=2, truncated_svd_n=2)

PANAS_df = x_df_nonan[['avg_weekly_PA', 'avg_weekly_NA','weighted_avg_weekly_PA',
                       'weighted_avg_weekly_NA', 'total_NA/PA', 'avg_weekly_NA/PA',
                       'weighted_avg_weekly_NA/PA', 'total_PA', 'total_NA']] #weekday, ID
PANAS_df = reduce_dimensionality(PANAS_df, pca_n=3, kernel_pca_n=2, truncated_svd_n=2)

y = np.array(y_df['HAMD']).reshape(-1,1)
all_x = np.array(reduced_x_df)
pca_x = np.array(reduced_x_df[['PCA_0', 'PCA_1']])
kernel_pca_x = np.array(reduced_x_df[['KernelPCA_0', 'KernelPCA_1']])
truncated_svd_x = np.array(reduced_x_df[['TruncatedSVD_0', 'TruncatedSVD_1']])
PANAS_short_x = np.array(PANAS_df[['avg_weekly_PA', 'avg_weekly_NA',
                                   'avg_weekly_NA/PA']])
PANAS_x = np.array(PANAS_df[['avg_weekly_PA', 'avg_weekly_NA',
                                 'weighted_avg_weekly_PA', 'weighted_avg_weekly_NA',
                                 'total_NA/PA', 'avg_weekly_NA/PA',
                                 'weighted_avg_weekly_NA/PA']])
PANAS_pca_x = np.array(PANAS_df[['PCA_0', 'PCA_1']])
PANAS_kernel_pca_x = np.array(PANAS_df[['KernelPCA_0', 'KernelPCA_1']])
PANAS_truncated_svd_x = np.array(PANAS_df[['TruncatedSVD_0', 'TruncatedSVD_1']])

impute(all_x, y, 'all data')
impute(pca_x, y, 'PCA')
impute(kernel_pca_x, y, 'Kenrel PCA')
impute(truncated_svd_x, y, 'Truncated SVD')
impute(PANAS_short_x, y, 'PANAS short')
impute(PANAS_x, y, 'PANAS')
impute(PANAS_pca_x, y, 'PANAS PCA')
impute(PANAS_kernel_pca_x, y, 'PANAS Kenrel PCA')
impute(PANAS_truncated_svd_x, y, 'PANAS Truncated SVD')

plt.show()
################################### LEGACY #################################
def legacy():
    HAMD = pd.read_csv(data_dir+'daily_survey_HAMD.csv')

    USER_N = 23


    input_df = HAMD[HAMD['group']=='MDD']

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(USER_N))


    def one_hot_id(series):
        b = label_binarizer.transform([int(series['ID'][1:])])
        series['ID_one_hot'] = b[0]
        series['has_PANAS'] = [int(not np.isnan(series['total_PA1'])), int(not np.isnan(series['total_PA2']))]
        return series

    df = input_df[input_df['HAMD'] >= 0]

    df = df.apply(one_hot_id, axis=1)
    # df = df.dropna()
    df = df.fillna(0)
    df = df.reset_index(drop = True)



    def create_PANAS_detailed_one_hot_missing():
        x1 = np.array(df[['total_PA1', 'total_NA1', 'total_PA2', 'total_NA2', 'total_PA', 'total_NA', 'avg_weekly_PA', 'avg_weekly_NA', 'weighted_avg_weekly_PA', 'weighted_avg_weekly_NA']])
        tmp_x2 = np.array(df[['ID_one_hot']])
        x2 = np.squeeze(np.array([list(row) for row in tmp_x2]))
        tmp_x3 = np.array(df[['has_PANAS']])
        x3 = np.squeeze(np.array([list(row) for row in tmp_x3]))
        x = np.hstack((x1, x2, x3))
        return x

    def create_PANAS_detailed_one_hot():
        x1 = np.array(df[['total_PA1', 'total_NA1', 'total_PA2', 'total_NA2', 'total_PA', 'total_NA', 'avg_weekly_PA', 'avg_weekly_NA', 'weighted_avg_weekly_PA', 'weighted_avg_weekly_NA']])
        tmp_x2 = np.array(df[['ID_one_hot']])
        x2 = np.squeeze(np.array([list(row) for row in tmp_x2]))
        x = np.hstack((x1, x2))
        return x

    def create_PANAS_overall_one_hot():
        x1 = np.array(df[['total_PA', 'total_NA', 'avg_weekly_PA', 'avg_weekly_NA', 'weighted_avg_weekly_PA', 'weighted_avg_weekly_NA']])
        tmp_x2 = np.array(df[['ID_one_hot']])
        x2 = np.squeeze(np.array([list(row) for row in tmp_x2]))
        x = np.hstack((x1, x2))
        return x

    def create_PANAS_overall():
        x1 = np.array(df[['total_PA', 'total_NA', 'avg_weekly_PA', 'avg_weekly_NA', 'weighted_avg_weekly_PA', 'weighted_avg_weekly_NA']])
        return x1

    def create_PANAS_ratio():
        x1 = np.array(df[['total_NA/PA']])
        return x1