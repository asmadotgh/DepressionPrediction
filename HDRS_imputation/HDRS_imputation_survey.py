import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from my_constants import *
from dimensionality_reduction import preprocess_survey_x_y, reduce_dimensionality


np.random.seed(SEED)

BEST_VALIDATION_RMSE = 1000
BEST_X = None
BEST_Y = None
BEST_TTL = None
BEST_MDL_NAME = None
BEST_MDL = None


def split_data_ind(inds):
    #TODO: not the first score, no two from the same person
    test_N = int(len(inds)*0.1)
    np.random.shuffle(inds)
    ind_train = inds[test_N:]
    ind_test = inds[0:test_N]
    return ind_train, ind_test

def split_data_old(x, y):
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

def loo_split_data(x):
    loo = LeaveOneOut()
    splits = loo.split(x)
    for train, test in splits:
        print("%s %s" % (train, test))

def impute(inp_x, inp_y, ttl, mdl, ind_train, ind_test):

    global BEST_VALIDATION_RMSE, BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL

    x = np.array(inp_x)[ind_train]
    y = np.array(inp_y)[ind_train]

    # Create linear regression object
    if mdl == 'regression':
        regr = linear_model.LinearRegression()
    elif mdl == 'ridge':
        regr = linear_model.RidgeCV(alphas=REGULARIZATION_ALPHAS)
    elif mdl == 'lasso':
        regr = linear_model.LassoCV(alphas=REGULARIZATION_ALPHAS)
    elif mdl == 'elasticNet':
        regr = linear_model.ElasticNetCV(alphas=REGULARIZATION_ALPHAS)
    elif mdl == 'theil':
        regr = linear_model.TheilSenRegressor(random_state=SEED)
    elif mdl == 'ransac':
        regr = linear_model.RANSACRegressor(random_state=SEED)
    elif mdl == 'huber':
        regr = linear_model.HuberRegressor(random_state=SEED)


    inds = range(len(y))

    loo = LeaveOneOut()
    splits = loo.split(inds)
    avg_train_RMSE = 0
    avg_validation_RMSE = 0
    for tmp_train_inds, validation_inds in splits:
        train_inds = list(tmp_train_inds)
        x_train = x[train_inds]
        y_train = y[train_inds]
        x_validation = x[validation_inds]
        y_validation = y[validation_inds]

        # Train the model using the training sets
        regr.fit(x_train, y_train)

        validation_RMSE = np.sqrt(mean_squared_error(y_validation, regr.predict(x_validation)))
        train_RMSE = np.sqrt(mean_squared_error(y_train, regr.predict(x_train)))

        avg_train_RMSE += train_RMSE
        avg_validation_RMSE += validation_RMSE

    avg_train_RMSE /= len(inds)
    avg_validation_RMSE /= len(inds)

    if avg_validation_RMSE < BEST_VALIDATION_RMSE:
        BEST_X = inp_x
        BEST_Y = inp_y
        BEST_TTL = ttl
        BEST_MDL_NAME = mdl
        BEST_MDL = regr
        BEST_VALIDATION_RMSE = avg_validation_RMSE

    print mdl+', '+ttl+', train RMSE: %f, validation RMSE: %f ' %(avg_train_RMSE, avg_validation_RMSE)

def legacy_impute_without_first_inds(x, y, ttl):
    global first_inds

    # Create linear regression object
    regr = linear_model.LinearRegression()

    inds = range(len(x))
    for i in first_inds:
        inds.remove(i)
    loo = LeaveOneOut()
    splits = loo.split(inds)
    avg_train_RMSE = 0
    avg_test_RMSE = 0
    for tmp_train_inds, test_inds in splits:
        train_inds = list(tmp_train_inds) + list(first_inds)
        x_train = x[train_inds]
        y_train = y[train_inds]
        x_test = x[test_inds]
        y_test = y[test_inds]

        # Train the model using the training sets
        regr.fit(x_train, y_train)

        test_RMSE = np.sqrt(mean_squared_error(y_test, regr.predict(x_test)))
        train_RMSE = np.sqrt(mean_squared_error(y_train, regr.predict(x_train)))

        avg_train_RMSE += train_RMSE
        avg_test_RMSE += test_RMSE

    avg_train_RMSE /= len(inds)
    avg_test_RMSE /= len(inds)

    print ttl+' ,train RMSE:%f, test RMSE: %f ' %(avg_train_RMSE, avg_test_RMSE)


def legacy_impute(x, y, ttl):
    ind_train, x_train, y_train, ind_test, x_test, y_test = split_data(x, y)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # print sklearn.feature_selection.f_regression(x, y, center=True)
    print regr.coef_
    print

    # err = 1.0-5.0/4.0*(1-r2_score(y_test, regr.predict(x_test), multioutput='variance_weighted'))

    test_RMSE = np.sqrt(mean_squared_error(y_test, regr.predict(x_test)))
    train_RMSE = np.sqrt(mean_squared_error(y_train, regr.predict(x_train)))
    print test_RMSE


    plt.figure(figsize=(16,4))

    plt.scatter(range(len(y)), y, label='HDRS', color = 'green', alpha=0.5)
    plt.scatter(ind_train, regr.predict(x_train), label='predicted HDRS - train', color = 'red', alpha=0.5)
    plt.scatter(ind_test, regr.predict(x_test), label='predicted HDRS - test', color = 'black', alpha=0.5)

    plt.title(ttl + ', train RMSE: '+'{:.3f}'.format(train_RMSE)+', test RMSE: '+'{:.3f}'.format(test_RMSE))
    plt.ylabel('HDRS')

    plt.legend(loc=2, scatterpoints=1)


def plot_prediction(x, y, ttl, mdl_name, mdl, validation_RMSE, ind_train, ind_test):

    test_RMSE = np.sqrt(mean_squared_error(np.array(y)[ind_test], mdl.predict(np.array(x)[ind_test])))

    plt.figure(figsize=(16, 4))
    plt.scatter(range(len(y)), y, label='HDRS', color='green', alpha=0.5)
    plt.scatter(ind_train, mdl.predict(np.array(x)[ind_train]), label='predicted HDRS - train', color='red', alpha=0.5)
    plt.scatter(ind_test, mdl.predict(np.array(x)[ind_test]), label='predicted HDRS - test', color='black', alpha=0.5)
    plt.title('model: '+mdl_name+', dataset: '+ttl +
              ', validation RMSE: '+'{:.3f}'.format(validation_RMSE)+
              ', test RMSE: '+'{:.3f}'.format(test_RMSE))
    plt.ylabel('HDRS')
    plt.legend(loc=2, scatterpoints=1)

    print 'model parameters: \n'
    print mdl.coef_

    plt.show()

all_df, x_df, y_df = preprocess_survey_x_y()

subset_df = all_df.dropna(subset=['HAMD'])
subset_df = subset_df[subset_df['day']>=0].reset_index(True)
first_inds = subset_df[subset_df['day']<14].index
x_df = subset_df.drop(['ID','HAMD'], inplace=False, axis=1)
y_df = subset_df[['ID', 'HAMD']]
x_df_nonan = x_df.fillna(0)
reduced_x_df, reduced_n = reduce_dimensionality(x_df_nonan, max_n=30, threshold=EXPLAINED_VARIANCE_THRESHOLD)

PANAS_df = x_df_nonan[['avg_weekly_PA','avg_weekly_NA','avg_weekly_NA/PA',
                       'weighted_avg_weekly_PA','weighted_avg_weekly_NA','weighted_avg_weekly_NA/PA',
                       'total_PA','total_NA','total_NA/PA',
                       'avg_overall_PA', 'avg_overall_NA','avg_overall_NA/PA',
                       'std_weekly_PA', 'std_weekly_NA',
                       'std_overall_PA', 'std_overall_NA']] #weekday, ID
reduced_PANAS_df, PANAS_reduced_n = reduce_dimensionality(PANAS_df, max_n=5, threshold=EXPLAINED_VARIANCE_THRESHOLD)

y = y_df[['HAMD']]#.reshape(-1,1)
all_x = reduced_x_df
pca_x = reduced_x_df[['PCA_'+str(i) for i in range(reduced_n)]]
kernel_pca_x = reduced_x_df[['KernelPCA_'+str(i) for i in range(reduced_n)]]
truncated_svd_x = reduced_x_df[['TruncatedSVD_'+str(i) for i in range(reduced_n)]]
PANAS_short_weekly_x = PANAS_df[['avg_weekly_PA', 'avg_weekly_NA', 'avg_weekly_NA/PA',
                                 'std_weekly_PA', 'std_weekly_NA']]
PANAS_short_weighted_weekly_x = PANAS_df[['weighted_avg_weekly_PA',
                       'weighted_avg_weekly_NA', 'weighted_avg_weekly_NA/PA']]
PANAS_short_daily_x = PANAS_df[['total_PA', 'total_NA', 'total_NA/PA']]
PANAS_x = PANAS_df
PANAS_pca_x = reduced_PANAS_df[['PCA_'+str(i) for i in range(PANAS_reduced_n)]]
PANAS_kernel_pca_x = reduced_PANAS_df[['KernelPCA_'+str(i) for i in range(PANAS_reduced_n)]]
PANAS_truncated_svd_x = reduced_PANAS_df[['TruncatedSVD_'+str(i) for i in range(PANAS_reduced_n)]]

# uncomment below for removing the first visits of each user
inds = range(len(y))
# for i in first_inds:
#         inds.remove(i)
ind_train, ind_test = split_data_ind(inds)
# ind_train = list(ind_train) + list(first_inds)
print '\ntrain indices:'
print ind_train
print '\ntest indices:'
print ind_test

# TODO: add models that are robust to outlier
# TODO: add hierarchical bayes
models = ['regression', 'ridge', 'lasso', 'elasticNet', 'theil', 'ransac', 'huber']
for mdl in models:
    impute(all_x, y, 'all data', mdl, ind_train, ind_test)
    impute(pca_x, y, 'PCA', mdl, ind_train, ind_test)
    impute(kernel_pca_x, y, 'Kernel PCA', mdl, ind_train, ind_test)
    impute(truncated_svd_x, y, 'Truncated SVD', mdl, ind_train, ind_test)
    impute(PANAS_short_weekly_x, y, 'PANAS short weekly', mdl, ind_train, ind_test)
    impute(PANAS_short_weighted_weekly_x, y, 'PANAS short weighted weekly', mdl, ind_train, ind_test)
    impute(PANAS_short_daily_x, y, 'PANAS short daily', mdl, ind_train, ind_test)
    impute(PANAS_x, y, 'PANAS', mdl, ind_train, ind_test)
    impute(PANAS_pca_x, y, 'PANAS PCA', mdl, ind_train, ind_test)
    impute(PANAS_kernel_pca_x, y, 'PANAS Kenrel PCA', mdl, ind_train, ind_test)
    impute(PANAS_truncated_svd_x, y, 'PANAS Truncated SVD', mdl, ind_train, ind_test)

plot_prediction(BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL, BEST_VALIDATION_RMSE, ind_train, ind_test)

#
# plt.show()
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