import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn import ensemble
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


def fill_nan(series):
    series['original'] = True
    if np.isnan(series['total_PA']):
        series['original'] = False
        series['total_PA'] = series['avg_overall_PA']
        series['total_NA'] = series['avg_overall_NA']
        series['total_NA/PA'] = series['total_NA']/series['total_PA']
    if np.isnan(series['avg_weekly_PA']):
        series['original'] = False
        series['avg_weekly_PA'] = series['avg_overall_PA']
        series['avg_weekly_NA'] = series['avg_overall_NA']
        series['avg_weekly_NA/PA'] = series['avg_overall_PA']/series['avg_overall_NA']
        series['weighted_avg_weekly_PA'] = series['avg_overall_PA']
        series['weighted_avg_weekly_NA'] = series['avg_overall_NA']
        series['weighted_avg_weekly_NA/PA'] = series['avg_overall_PA']/series['avg_overall_NA']
        series['std_weekly_PA'] = series['std_overall_PA']
        series['std_weekly_NA'] = series['std_overall_NA']
    return series


def split_data_ind(inds, test_N):
    np.random.shuffle(inds)
    ind_train = inds[test_N:]
    ind_test = inds[0:test_N]
    return ind_train, ind_test


def loo_split_data(x):
    loo = LeaveOneOut()
    splits = loo.split(x)
    for train, test in splits:
        print("%s %s" % (train, test))


def impute(inp_x, inp_y, ttl, mdl, ind_train, ind_test, model_file):

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
        regr = linear_model.RANSACRegressor(random_state=SEED, min_samples=0.2)
    elif mdl == 'huber':
        regr = linear_model.HuberRegressor(epsilon=2.0)
    elif mdl == 'adaBoost':
        regr = ensemble.GradientBoostingRegressor(random_state=SEED)
    elif mdl=='gb':
        regr = ensemble.AdaBoostRegressor(random_state=SEED)
    elif 'rf' in mdl: #rf_{n}
        n = int(mdl[mdl.find('_')+1:])
        regr = ensemble.RandomForestRegressor(random_state=SEED, n_estimators=n)


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

        validation_RMSE = np.sqrt(mean_squared_error(y_validation, np.round(regr.predict(x_validation))))
        train_RMSE = np.sqrt(mean_squared_error(y_train, np.round(regr.predict(x_train))))

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
    model_file.write(mdl+', '+ttl+', train RMSE: %f, validation RMSE: %f \n' %(avg_train_RMSE, avg_validation_RMSE))


def plot_prediction(x, y, ttl, mdl_name, mdl, validation_RMSE, ind_train, ind_test):

    test_RMSE = np.sqrt(mean_squared_error(np.array(y)[ind_test], np.round(mdl.predict(np.array(x)[ind_test]))))

    plt.figure(figsize=(16, 4))
    plt.scatter(range(len(y)), y, label='HDRS', color='green', alpha=0.5)
    plt.scatter(ind_train, np.round(mdl.predict(np.array(x)[ind_train])), label='predicted - train', color='red', alpha=0.5)
    plt.scatter(ind_test, np.round(mdl.predict(np.array(x)[ind_test])), label='predicted - test', color='black', alpha=0.5)
    plt.title('model: '+mdl_name+', dataset: '+ttl +
              ', validation RMSE: '+'{:.3f}'.format(validation_RMSE)+
              ', test RMSE: '+'{:.3f}'.format(test_RMSE))
    plt.ylabel('HDRS')
    plt.legend(loc=2, scatterpoints=1)

    model_file = open(MODEL_FILE, "a+")
    model_file.write('\nBest Model: '+mdl_name+', '+ttl+', validation RMSE: %f, test RMSE: %f \n' %(validation_RMSE, test_RMSE))
    if mdl != 'ransac' and mdl != 'gb' and mdl != 'adaBoost' and 'rf' not in mdl:
        print 'model parameters: \n'
        print mdl.coef_
        model_file.write('coefficients:\n')
        for item in mdl.coef_:
            model_file.write('%s \t' % item)
        model_file.write('\n')
    model_file.close()

    plt.show()


def impute_using_best_model(df, ttl, mdl_name, mdl):
    if ttl != 'PANAS':
        print 'NOT PANAS'
        return
    df = df.apply(is_imputed, axis=1)
    def fill_HAMD(series):
        x = series[['avg_weekly_PA','avg_weekly_NA','avg_weekly_NA/PA',
                    'weighted_avg_weekly_PA','weighted_avg_weekly_NA','weighted_avg_weekly_NA/PA',
                    'total_PA','total_NA','total_NA/PA',
                    'avg_overall_PA', 'avg_overall_NA','avg_overall_NA/PA',
                    'std_weekly_PA', 'std_weekly_NA',
                    'std_overall_PA', 'std_overall_NA']]
        y = series['HAMD']
        if np.isnan(y):
            series['HAMD'] = np.round(mdl.predict(x.reshape(1, -1))).flatten()[0]
        return series
    df = df.apply(fill_HAMD, axis=1)

    df[['date', 'ID', 'HAMD', 'imputed']].to_csv(data_dir+'HAMD_imputed_survey.csv', index=False)

    return

all_df, x_df, y_df = preprocess_survey_x_y()


subset_df = all_df.dropna(subset=['HAMD'])
subset_df = subset_df[subset_df['ID']!='M005']
subset_df = subset_df[subset_df['day']>=0].reset_index(True)
first_inds = subset_df[subset_df['day']<14].index
x_df = subset_df.drop(['ID','HAMD', 'date'], inplace=False, axis=1)
y_df = subset_df[['ID', 'HAMD']]
x_df_nonan = x_df
x_df_nonan = x_df_nonan.apply(fill_nan, axis=1)
non_original_inds = x_df_nonan[x_df_nonan['original']==False].index
x_df_nonan.drop(['original'], inplace=True, axis=1)
x_df_nonan = x_df_nonan.fillna(0)

reduced_x_df, reduced_n = reduce_dimensionality(x_df_nonan, max_n=30, threshold=EXPLAINED_VARIANCE_THRESHOLD)

PANAS_df = x_df_nonan[['avg_weekly_PA','avg_weekly_NA','avg_weekly_NA/PA',
                       'weighted_avg_weekly_PA','weighted_avg_weekly_NA','weighted_avg_weekly_NA/PA',
                       'total_PA','total_NA','total_NA/PA',
                       'avg_overall_PA', 'avg_overall_NA','avg_overall_NA/PA',
                       'std_weekly_PA', 'std_weekly_NA',
                       'std_overall_PA', 'std_overall_NA']] #weekday, ID
reduced_PANAS_df, PANAS_reduced_n = reduce_dimensionality(PANAS_df, max_n=5, threshold=EXPLAINED_VARIANCE_THRESHOLD)

y = y_df[['HAMD']]#.reshape(-1,1)
all_x = x_df_nonan
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
for i in non_original_inds:
         inds.remove(i)
ind_train, ind_test = split_data_ind(inds, int(TEST_RATIO*len(y)))
# ind_train = list(ind_train) + list(first_inds)
ind_train = list(ind_train) + list(non_original_inds)

print '\n dataset size:'
print len(y)
print '\ntrain indices:'
print ind_train
print '\ntest indices:'
print ind_test

# regression, regularized versions, models that are robust to outliers
# TODO: add hierarchical bayes
models = ['regression', 'ridge', 'lasso', 'elasticNet', 'huber', 'ransac', 'theil',
          'adaBoost', 'gb']
#adding rf models
n_estimators = [5, 10, 15, 20, 25]
for n in n_estimators:
    models.append('rf_'+str(n))
model_file = open(MODEL_FILE, "w")
model_file.close()
for mdl in models:
    model_file = open(MODEL_FILE, "a+")
    impute(all_x, y, 'all data', mdl, ind_train, ind_test, model_file)
    impute(pca_x, y, 'PCA', mdl, ind_train, ind_test, model_file)
    impute(kernel_pca_x, y, 'Kernel PCA', mdl, ind_train, ind_test, model_file)
    impute(truncated_svd_x, y, 'Truncated SVD', mdl, ind_train, ind_test, model_file)
    impute(PANAS_short_weekly_x, y, 'PANAS short weekly', mdl, ind_train, ind_test, model_file)
    impute(PANAS_short_weighted_weekly_x, y, 'PANAS short weighted weekly', mdl, ind_train, ind_test, model_file)
    impute(PANAS_short_daily_x, y, 'PANAS short daily', mdl, ind_train, ind_test, model_file)
    impute(PANAS_x, y, 'PANAS', mdl, ind_train, ind_test, model_file)
    impute(PANAS_pca_x, y, 'PANAS PCA', mdl, ind_train, ind_test, model_file)
    impute(PANAS_kernel_pca_x, y, 'PANAS Kernel PCA', mdl, ind_train, ind_test, model_file)
    impute(PANAS_truncated_svd_x, y, 'PANAS Truncated SVD', mdl, ind_train, ind_test, model_file)
    model_file.close()


plot_prediction(BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL, BEST_VALIDATION_RMSE, ind_train, ind_test)

impute_df = all_df[all_df['ID']!='M005']
impute_df = impute_df[impute_df['day']>=0].reset_index(True)
impute_df = impute_df.apply(fill_nan, axis=1)
impute_df.drop(['original'], inplace=True, axis=1)

impute_using_best_model(impute_df, BEST_TTL, BEST_MDL_NAME, BEST_MDL)