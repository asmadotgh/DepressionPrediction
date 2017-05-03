from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import ensemble
from sklearn import gaussian_process
import numpy as np
import pandas as pd
from my_constants import *
from dimensionality_reduction import reduce_dimensionality
from sklearn.model_selection import KFold


np.random.seed(SEED)

BEST_VALIDATION_RMSE = 1000
BEST_X = None
BEST_Y = None
BEST_TTL = None
BEST_MDL_NAME = None
BEST_MDL = None



def split_data_ind(inds, test_N):
    np.random.shuffle(inds)
    ind_train = inds[test_N:]
    ind_test = inds[0:test_N]
    return ind_train, ind_test

def predict(inp_x, inp_y, ttl, mdl, ind_train, ind_test, model_file):

    global BEST_VALIDATION_RMSE, BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL

    x = np.array(inp_x)[ind_train]
    y = np.array(inp_y)[ind_train]

    # Create linear regression object
    if mdl == 'regression':
        regr = linear_model.LinearRegression()
    elif mdl == 'ridge':
        regr = linear_model.RidgeCV(cv=K_FOLD_N, alphas=REGULARIZATION_ALPHAS)
    elif mdl == 'lasso':
        regr = linear_model.LassoCV(cv=K_FOLD_N, alphas=REGULARIZATION_ALPHAS)
    elif mdl == 'elasticNet':
        regr = linear_model.ElasticNetCV(cv=K_FOLD_N, alphas=REGULARIZATION_ALPHAS)
    elif mdl == 'theil':
        regr = linear_model.TheilSenRegressor(random_state=SEED)
    elif 'ransac' in mdl: #ransac_{ms}
        ms = float(mdl[mdl.find('_')+1:])
        regr = linear_model.RANSACRegressor(random_state=SEED, min_samples=ms)
    elif 'huber' in mdl: #huber_e{eps}_a{al}
        eps = float(mdl[mdl.find('_e')+2:mdl.find('_a')])
        al = float(mdl[mdl.find('_a')+2:])
        regr = linear_model.HuberRegressor(epsilon=eps, alpha=al)
    elif 'rf' in mdl: #rf_{n}
        n = int(mdl[mdl.find('_')+1:])
        regr = ensemble.RandomForestRegressor(random_state=SEED, n_estimators=n)
    # elif mdl == 'gp':
    #     regr = gaussian_process.GaussianProcessRegressor(n_restarts_optimizer=N_RESTARTS_OPTIMIZER, random_state=SEED)

    inds = range(len(y))
    kf = KFold(n_splits=K_FOLD_N)
    splits = kf.split(inds)
    avg_train_RMSE = 0
    avg_validation_RMSE = 0

    for train_inds, validation_inds in splits:
        x_train = x[train_inds]
        y_train = y[train_inds]
        x_validation = x[validation_inds]
        y_validation = y[validation_inds]

        # Train the model using the training sets
        try:
            regr.fit(x_train, y_train)

            validation_RMSE = np.sqrt(mean_squared_error(y_validation, np.round(regr.predict(x_validation))))
            train_RMSE = np.sqrt(mean_squared_error(y_train, np.round(regr.predict(x_train))))

            avg_train_RMSE += train_RMSE
            avg_validation_RMSE += validation_RMSE
        except:
            print 'not converged'

    avg_train_RMSE /= K_FOLD_N
    avg_validation_RMSE /= K_FOLD_N

    if avg_validation_RMSE < BEST_VALIDATION_RMSE:
        BEST_X = inp_x
        BEST_Y = inp_y
        BEST_TTL = ttl
        BEST_MDL_NAME = mdl
        BEST_MDL = regr
        BEST_VALIDATION_RMSE = avg_validation_RMSE

    print mdl+', '+ttl+', train RMSE: %f, validation RMSE: %f ' %(avg_train_RMSE, avg_validation_RMSE)
    model_file.write(mdl+', '+ttl+', train RMSE: %f, validation RMSE: %f \n' %(avg_train_RMSE, avg_validation_RMSE))


def plot_prediction(x, y, ttl, mdl_name, mdl, validation_RMSE, ind_train, ind_test, HAMD_file):
    MODEL_FILE_NAME = MODEL_FILE[0:-4] + '_' +HAMD_file[0:-4]+'.txt'

    test_RMSE = np.sqrt(mean_squared_error(np.array(y)[ind_test], np.round(mdl.predict(np.array(x)[ind_test]))))

    plt.figure(figsize=(16, 4))
    plt.scatter(range(len(y)), y, label='HDRS', color='green', alpha=0.5)
    plt.scatter(ind_train, np.round(mdl.predict(np.array(x)[ind_train])), label='predicted - train', color='red', alpha=0.5)
    plt.scatter(ind_test, np.round(mdl.predict(np.array(x)[ind_test])), label='predicted - test', color='black', alpha=0.5)
    plt.title('imputation: '+ HAMD_file[0:-4]+', model: '+mdl_name+', dataset: '+ttl +
              ', validation RMSE: '+'{:.3f}'.format(validation_RMSE)+
              ', test RMSE: '+'{:.3f}'.format(test_RMSE))
    plt.ylabel('HDRS')
    plt.legend(loc=2, scatterpoints=1)

    model_file = open(MODEL_FILE_NAME, "a+")
    model_file.write('\nBest Model: '+mdl_name+', '+ttl+', validation RMSE: %f, test RMSE: %f \n' %(validation_RMSE, test_RMSE))
    if mdl_name == 'gp' or 'ransac' in mdl_name or 'rf' in mdl_name:
        print 'no coefficiants to print for this model.'
    else:
        print 'model parameters: \n'
        print mdl.coef_
        model_file.write('coefficients:\n')
        for item in mdl.coef_:
            model_file.write('%s \t' % item)
        model_file.write('\n')
    model_file.close()
    fig_title = HAMD_file[0:-4]+'_'+mdl_name+'_'+ttl+\
    '_v_'+'{:.3f}'.format(validation_RMSE)+\
    '_t_'+'{:.3f}'.format(test_RMSE)+'.pdf'
    plt.savefig('figs/'+fig_title, transparent=True, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(16, 4))
    plt.scatter(range(len(ind_test)), np.array(y)[ind_test], label='HDRS', color='green', alpha=0.5)
    plt.scatter(range(len(ind_test)), np.round(mdl.predict(np.array(x)[ind_test])), label='predicted - test', color='black', alpha=0.5)
    plt.title('imputation: '+ HAMD_file[0:-4]+', model: '+mdl_name+', dataset: '+ttl +
              ', validation RMSE: '+'{:.3f}'.format(validation_RMSE)+
              ', test RMSE: '+'{:.3f}'.format(test_RMSE))
    plt.ylabel('HDRS')
    plt.legend(loc=2, scatterpoints=1)
    plt.savefig('figs/test/'+fig_title, transparent=True, format='pdf', bbox_inches='tight')


def run_prediction(HAMD_file):
    MODEL_FILE_NAME = MODEL_FILE[0:-4] + '_' +HAMD_file[0:-4]+'_robust.txt'
    all_df = pd.read_csv(data_dir+HAMD_file)
    feature_df = pd.read_csv(feature_dir+'daily_all.csv')
    all_df = all_df.merge(feature_df, on=['ID', 'date'], how='outer')
    all_df = all_df.dropna(subset=['HAMD'])
    all_df = convert_one_hot_str(all_df, 'ID')

    y_df = all_df[['ID', 'HAMD', 'date', 'imputed']]
    x_df = all_df.drop(['ID','HAMD','date', 'imputed'], inplace=False, axis=1)
    x_df_nonan = x_df.fillna(0)

    remove_col = []
    for i in range(len(x_df_nonan.columns.values)):
        if np.std(x_df_nonan[x_df_nonan.columns[i]])==0:
            remove_col.append(i)
    x_df_nonan = x_df_nonan.drop(x_df_nonan.columns[remove_col], axis=1)


    reduced_x_df, reduced_n = reduce_dimensionality(x_df_nonan, max_n=25, threshold=EXPLAINED_VARIANCE_THRESHOLD)

    y = y_df[['HAMD']]
    all_x = x_df_nonan
    pca_x = reduced_x_df[['PCA_'+str(i) for i in range(reduced_n)]]
    kernel_pca_x = reduced_x_df[['KernelPCA_'+str(i) for i in range(reduced_n)]]
    truncated_svd_x = reduced_x_df[['TruncatedSVD_'+str(i) for i in range(reduced_n)]]

    all_columns = x_df_nonan.columns.values
    sub_columns = []
    for col in all_columns:
        if 'sleep' in col: #'daily' in col or
            sub_columns.append(col)
    #sub_x = x_df_nonan[sub_columns]
    sub_x = x_df_nonan[['call_daily_IncomingDismissed_count_call',
                        'call_daily_IncomingMissed_count_call',
                        'call_daily_Incoming_count_call',
                        'call_daily_Incoming_mean_call_duration',
                        'call_daily_Incoming_median_call_duration',
                        'call_daily_Incoming_std_call_duration',
                        'call_daily_Incoming_sum_call_duration',
                        'call_daily_Outgoing_count_call',
                        'call_daily_Outgoing_mean_call_duration',
                        'call_daily_Outgoing_median_call_duration',
                        'call_daily_Outgoing_std_call_duration',
                        'call_daily_Outgoing_sum_call_duration',
                        'call_daily_incoming_outgoing_call_duration',
                        'call_daily_incoming_outgoing_call_count',
                        'display_daily_sum_on_duration',
                        'display_daily_std_on_duration',
                        'display_daily_mean_on_duration',
                        'display_daily_median_on_duration',
                        'display_daily_count_on',
                        'location_daily_count',
                        'location_daily_total_std',
                        'sleep_24hrs_fraction_recording',
                        'sleep_24hrs_sleep_(s)',
                        'sleep_night_sleep_(s)',
                        'sleep_night_fraction_recording',
                        'sleep_night_sleep_onset_timeelapsed_since_noon_(s)',
                        'sleep_night_max_uninterrupted_sleep_(s)',
                        'sleep_night_nbwakeups',
                        'sleep_ day_wakeup_onset_timeelapsed_since_midnight_(s)',
                        'sleep_sleep_reg_index',
                        'sms_daily_Incoming_count_sms',
                        'ID_M004', 'ID_M006', 'ID_M008', 'ID_M011', 'ID_M012', 'ID_M013',
                        'ID_M015', 'ID_M016', 'ID_M017', 'ID_M020', 'ID_M022']]

    reduced_sub_x_df, reduced_sub_n = reduce_dimensionality(sub_x, max_n=25, threshold=EXPLAINED_VARIANCE_THRESHOLD)
    pca_sub_x = reduced_sub_x_df[['PCA_'+str(i) for i in range(reduced_sub_n)]]
    kernel_pca_sub_x = reduced_sub_x_df[['KernelPCA_'+str(i) for i in range(reduced_sub_n)]]
    truncated_svd_sub_x = reduced_sub_x_df[['TruncatedSVD_'+str(i) for i in range(reduced_sub_n)]]


    inds = range(len(y))
    imputed_inds = y_df[y_df['imputed']=='y'].index
    for i in imputed_inds:
        inds.remove(i)
    #TODO: do we have enough tests?
    ind_train, ind_test = split_data_ind(inds, int(TEST_RATIO*len(y)))
    ind_train = list(ind_train) + list(imputed_inds)

    print '\n dataset size:'
    print len(y)
    print '\ntrain indices:'
    print ind_train
    print '\ntest indices:'
    print ind_test

    models = ['theil']

    # adding ransac models
    min_samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    for ms in min_samples:
        models.append('ransac_'+str(ms))

    # adding huber models
    epsilons = [1.0, 1.35, 1.5]
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    for eps in epsilons:
        for al in alphas:
            models.append('huber_e'+str(eps)+'_a'+str(al))

    model_file = open(MODEL_FILE_NAME, "w")
    model_file.close()
    for mdl in models:
        model_file = open(MODEL_FILE_NAME, "a+")
        predict(all_x, y, 'all data', mdl, ind_train, ind_test, model_file)
        predict(pca_x, y, 'PCA', mdl, ind_train, ind_test, model_file)
        predict(kernel_pca_x, y, 'Kernel PCA', mdl, ind_train, ind_test, model_file)
        predict(truncated_svd_x, y, 'Truncated SVD', mdl, ind_train, ind_test, model_file)
        predict(sub_x, y, 'sub data', mdl, ind_train, ind_test, model_file)
        predict(pca_sub_x, y, 'PCA sub', mdl, ind_train, ind_test, model_file)
        predict(kernel_pca_sub_x, y, 'Kernel PCA sub', mdl, ind_train, ind_test, model_file)
        predict(truncated_svd_sub_x, y, 'Truncated SVD sub', mdl, ind_train, ind_test, model_file)
        model_file.close()

    plot_prediction(BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL, BEST_VALIDATION_RMSE, ind_train, ind_test, HAMD_file)


HAMD_files = ['HAMD_imputed_survey.csv']
# HAMD_files = ['HAMD_imputed_survey.csv',
#               'HAMD_original.csv',
#               'HAMD_imputed_linear.csv']

for HAMD_file in HAMD_files:
    BEST_VALIDATION_RMSE = 1000
    BEST_X = None
    BEST_Y = None
    BEST_TTL = None
    BEST_MDL_NAME = None
    BEST_MDL = None
    run_prediction(HAMD_file)


#TODO sub feature selection

#TODO: GP regression (in the other file)

#TODO: encode missing data?

#TODO hieriarchichal bayes with STAN

#TODO: sensor high dimentional featurs -> NN -> new features