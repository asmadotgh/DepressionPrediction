from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from my_constants import *
from dimensionality_reduction import reduce_dimensionality
from sklearn.model_selection import KFold
from scipy import spatial

np.random.seed(SEED)

BEST_VALIDATION_RMSE = 1000000
BEST_Y = None
BEST_Y_PREDICTION = None
BEST_MDL_NAME = None



def get_knn_inds(K, sample, inp_x, train_inds):
    x = inp_x[train_inds]
    tree = spatial.KDTree(data=x)
    tree.data
    dist, indexes = tree.query(x=sample, k=K, )#(x, k=1, eps=0, p=2, distance_upper_bound=inf)
    if K==1:
        return list([indexes])
    else:
        return list(indexes)

def predict(xs, inp_sr_ys, inp_y, mdl, ind_train, ind_test, model_file):

    global BEST_VALIDATION_RMSE, BEST_Y, BEST_Y_PREDICTION, BEST_MDL_NAME

    kernel_pca_sub_x = np.array(xs[6])
    kernel_pca_sub_x_2 = np.array(xs[10])
    sub_x = np.array(xs[4])

    #order of datasets used: ['kernel PCA sub', 'kernel PCA sub hist', 'sub data', 'kernel PCA sub hist']

    y = np.array(inp_y['HAMD'])[ind_train]
    sr_ys = inp_sr_ys[ind_train]

    inds = range(len(y))
    kf = KFold(n_splits=K_FOLD_N)
    splits = kf.split(inds)

    avg_train_RMSE = 0
    avg_validation_RMSE = 0

    for train_inds, validation_inds in splits:

        ensemble_y = []
        for i in range(len(inp_y)):
            if mdl == 'avg':
                ensemble_y.append(np.round(np.mean(inp_sr_ys[i])))
            elif mdl =='median':
                ensemble_y.append(np.median(inp_sr_ys[i]))
            elif 'ensemble' in mdl:
                k = int(mdl[mdl.find('_')+1:])
                if i in ind_train:
                    tmp_k = k+1
                else:
                    tmp_k = k
                knn_inds_sub_x = get_knn_inds(tmp_k, sub_x[i], sub_x[ind_train], train_inds)
                knn_inds_kernel_pca_sub_x = get_knn_inds(tmp_k, kernel_pca_sub_x[i], kernel_pca_sub_x[ind_train], train_inds)
                knn_inds_kernel_pca_sub_x_2 = get_knn_inds(tmp_k, kernel_pca_sub_x_2[i], kernel_pca_sub_x_2[ind_train], train_inds)
                if i in ind_train:
                    knn_inds_sub_x = knn_inds_sub_x[1:]
                    knn_inds_kernel_pca_sub_x = knn_inds_kernel_pca_sub_x[1:]
                    knn_inds_kernel_pca_sub_x_2 = knn_inds_kernel_pca_sub_x_2[1:]
                best_j_RMSE = 1000
                best_j_ind = None
                for j in range(np.shape(inp_sr_ys)[1]):
                    if j == 0:
                        knn_inds = knn_inds_kernel_pca_sub_x
                    elif j == 1 or j == 3:
                        knn_inds = knn_inds_kernel_pca_sub_x_2
                    elif j == 2:
                        knn_inds = knn_inds_sub_x
                    # print knn_inds
                    j_RMSE = np.sqrt(mean_squared_error(y[knn_inds], sr_ys[knn_inds, j]))
                    if j_RMSE < best_j_RMSE:
                        best_j_RMSE = j_RMSE
                        best_j_ind = j
                ensemble_y.append(inp_sr_ys[i, best_j_ind])
                # print 'ind: '+str(best_j_ind) + ', RMSE: '+str(best_j_RMSE)

        ensemble_y = np.array(ensemble_y)

        y_train = y[train_inds]
        y_validation = y[validation_inds]
        ensemble_y_train = ensemble_y[train_inds]
        ensemble_y_validation = ensemble_y[validation_inds]

        validation_RMSE = np.sqrt(mean_squared_error(y_validation, ensemble_y_validation))
        train_RMSE = np.sqrt(mean_squared_error(y_train, ensemble_y_train))

        avg_train_RMSE += train_RMSE
        avg_validation_RMSE += validation_RMSE

    avg_train_RMSE /= K_FOLD_N
    avg_validation_RMSE /= K_FOLD_N

    if avg_validation_RMSE < BEST_VALIDATION_RMSE:
        BEST_MDL_NAME = mdl
        BEST_Y = inp_y
        BEST_Y_PREDICTION = ensemble_y
        BEST_VALIDATION_RMSE = avg_validation_RMSE


    print mdl+', train RMSE: %f, validation RMSE: %f ' %(avg_train_RMSE, avg_validation_RMSE)
    model_file.write(mdl+', train RMSE: %f, validation RMSE: %f \n' %(avg_train_RMSE, avg_validation_RMSE))


def plot_prediction(y, y_pred, mdl_name, validation_RMSE, ind_train, ind_test, HAMD_file):
    MODEL_FILE_NAME = MODEL_FILE[0:-4] + '_' +HAMD_file[0:-4]+'_ensemble.txt'

    test_RMSE = np.sqrt(mean_squared_error(np.array(y)[ind_test], y_pred[ind_test]))

    plt.figure(figsize=(16, 4))
    plt.scatter(range(len(y)), y, label='HDRS', color='green', alpha=0.5)
    plt.scatter(ind_train, y_pred[ind_train], label='predicted - train', color='red', alpha=0.5)
    plt.scatter(ind_test, y_pred[ind_test], label='predicted - test', color='black', alpha=0.5)
    plt.title('imputation: '+ HAMD_file[0:-4]+', model: '+mdl_name +
              ', validation RMSE: '+'{:.3f}'.format(validation_RMSE)+
              ', test RMSE: '+'{:.3f}'.format(test_RMSE))
    plt.ylabel('HDRS')
    plt.legend(loc=2, scatterpoints=1)

    model_file = open(MODEL_FILE_NAME, "a+")
    model_file.write('\nBest Model: '+mdl_name+', validation RMSE: %f, test RMSE: %f \n' %(validation_RMSE, test_RMSE))
    model_file.close()
    fig_title = HAMD_file[0:-4]+'_'+mdl_name+\
    '_v_'+'{:.3f}'.format(validation_RMSE)+\
    '_t_'+'{:.3f}'.format(test_RMSE)+'.pdf'
    plt.savefig('figs/'+fig_title, transparent=True, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(16, 4))
    plt.scatter(range(len(ind_test)), np.array(y)[ind_test], label='HDRS', color='green', alpha=0.5)
    plt.scatter(range(len(ind_test)), y_pred[ind_test], label='predicted - test', color='black', alpha=0.5)
    plt.title('imputation: '+ HAMD_file[0:-4]+', model: '+mdl_name+
              ', validation RMSE: '+'{:.3f}'.format(validation_RMSE)+
              ', test RMSE: '+'{:.3f}'.format(test_RMSE))
    plt.ylabel('HDRS')
    plt.legend(loc=2, scatterpoints=1)
    plt.savefig('figs/test/'+fig_title, transparent=True, format='pdf', bbox_inches='tight')


def run_prediction(HAMD_file):
    MODEL_FILE_NAME = MODEL_FILE[0:-4] + '_' +HAMD_file[0:-4]+'_ensemble.txt'
    all_df = pd.read_csv(data_dir+HAMD_file)
    feature_df = pd.read_csv(feature_dir+'daily_all.csv')
    all_df = all_df.merge(feature_df, on=['ID', 'date'], how='outer')
    all_df = all_df.dropna(subset=['HAMD'])
    all_df = convert_one_hot_str(all_df, 'ID')

    y_df = all_df[['ID', 'HAMD', 'date', 'imputed']]
    single_regressors = ['basic', 'robust', 'rf', 'gp']
    for sr in single_regressors:
        tmp_y_df = pd.read_csv(results_dir+sr+'.csv', index_col=0)
        y_df = y_df.join(tmp_y_df)

    sr_ys = np.array(y_df.drop(['ID', 'date', 'imputed', 'HAMD'], inplace=False, axis=1))
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
    sub_x = x_df_nonan[SUB_FEATURES]


    reduced_sub_x_df, reduced_sub_n = reduce_dimensionality(sub_x, max_n=25, threshold=EXPLAINED_VARIANCE_THRESHOLD)
    pca_sub_x = reduced_sub_x_df[['PCA_'+str(i) for i in range(reduced_sub_n)]]
    kernel_pca_sub_x = reduced_sub_x_df[['KernelPCA_'+str(i) for i in range(reduced_sub_n)]]
    truncated_svd_sub_x = reduced_sub_x_df[['TruncatedSVD_'+str(i) for i in range(reduced_sub_n)]]

    sub_x_prev_day = sub_x.shift(periods=1)
    sub_x_prev_day.iloc[0] = sub_x_prev_day.iloc[1]
    sub_x_prev_day.drop(ONE_HOT_USERS, inplace=True, axis=1)
    cols = sub_x_prev_day.columns.values
    sub_x_prev_day.columns = [col+'_hist' for col in cols]
    sub_x_2 = sub_x.join(sub_x_prev_day)

    reduced_sub_x_2_df, reduced_sub_n_2 = reduce_dimensionality(sub_x_2, max_n=30, threshold=EXPLAINED_VARIANCE_THRESHOLD)
    pca_sub_x_2 = reduced_sub_x_2_df[['PCA_'+str(i) for i in range(reduced_sub_n_2)]]
    kernel_pca_sub_x_2 = reduced_sub_x_2_df[['KernelPCA_'+str(i) for i in range(reduced_sub_n_2)]]
    truncated_svd_sub_x_2 = reduced_sub_x_2_df[['TruncatedSVD_'+str(i) for i in range(reduced_sub_n_2)]]


    inds = range(len(y))
    imputed_inds = y_df[y_df['imputed']=='y'].index
    for i in imputed_inds:
        inds.remove(i)
    ind_train, ind_test = split_data_ind(inds, int(TEST_RATIO*len(y)))
    ind_train = list(ind_train) + list(imputed_inds)

    print '\n dataset size:'
    print len(y)
    print '\ntrain indices:'
    print ind_train
    print '\ntest indices:'
    print ind_test



    models = ['avg', 'median']
    ks = [1, 5, 10, 20, 50, 75, 100]
    for k in ks:
        models.append('ensemble_'+str(k))

    model_file = open(MODEL_FILE_NAME, "w")
    model_file.close()

    xs = [all_x, pca_x, kernel_pca_x, truncated_svd_x,
          sub_x, pca_sub_x, kernel_pca_sub_x, truncated_svd_sub_x,
          sub_x_2, pca_sub_x_2, kernel_pca_sub_x_2, truncated_svd_sub_x_2]
    for mdl in models:
        model_file = open(MODEL_FILE_NAME, "a+")

        predict(xs, sr_ys, y, mdl, ind_train, ind_test, model_file)

        model_file.close()

    plot_prediction(BEST_Y, BEST_Y_PREDICTION, BEST_MDL_NAME, BEST_VALIDATION_RMSE, ind_train, ind_test, HAMD_file)


HAMD_files = ['HAMD_imputed_survey.csv']
# HAMD_files = ['HAMD_imputed_survey.csv',
#               'HAMD_original.csv',
#               'HAMD_imputed_linear.csv']

for HAMD_file in HAMD_files:
    BEST_VALIDATION_RMSE = 100000
    BEST_Y = None
    BEST_Y_PREDICTION = None
    BEST_MDL_NAME = None
    run_prediction(HAMD_file)


#TODO encode missing data?

#TODO hieriarchichal bayes with STAN

#TODO sensor high dimentional featurs -> NN -> new features