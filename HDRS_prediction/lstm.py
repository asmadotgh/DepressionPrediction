from sklearn.metrics import mean_squared_error
from dimensionality_reduction import reduce_dimensionality
import tensorflow as tf
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from my_constants import *


np.random.seed(SEED)

BEST_VALIDATION_RMSE = 1000
BEST_X = None
BEST_Y = None
BEST_TTL = None
BEST_MDL_NAME = None
BEST_MDL = None


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
    if mdl_name != 'ransac' and mdl_name != 'gp':
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

    plt.show()



class LSTMBaseline:
    def __init__(self, hdim):
            self.n_hidden = hdim  # 1st layer number of features
            self.weights = {
                'h1_tr': tf.Variable(tf.random_normal([self.n_hidden, 1], mean=0.0, stddev=0.01))
            }
            self.biases = {
                'b1_tr': tf.Variable(tf.random_normal([1], mean=0.0, stddev=0.01))
            }
            self.cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True)

    def network(self, x, dt, y, rnn_cell, N, T):
        # dt = tf.concat(1, [tf.ones([N, 1]), dt])
        x_t_y_concat = tf.concat(2, [x, dt])
        x_t_y_concat = tf.concat(2, [x_t_y_concat, y])
        lstm_val, state = tf.nn.dynamic_rnn(rnn_cell, x_t_y_concat, dtype=tf.float32)
        out_layer = tf.add(tf.matmul(tf.reshape(lstm_val, [N * T, -1]), self.weights['h1_tr']), self.biases['b1_tr'])
        out_layer = tf.reshape(out_layer, [N, T, -1])
        return out_layer


def predict(x_df, y_df, ttl):

    tf.reset_default_graph()


    EXP = 'LSTMRegression'
    SEED = 1
    learning_rate = 0.001
    hdim = 100   # H
    DATASET = 'MIMIC_BP'#'MIMIC_glucose'
    N = 1           # mini batch size
    T = 50          # time series length
    niters = 20 #50


    dir_path = 'LSTM/'+ttl+'/'
    figs_path = dir_path + 'figs/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    train_x = np.reshape(np.array(x_df[0:9*N*T]), (9*N, T, -1))
    train_y = np.reshape(np.array(y_df[0:9*N*T]), (9*N, T, -1))

    F = np.shape(train_x)[2]
    S1_train = np.shape(train_x)[0]
    S2_train = np.shape(train_x)[1]
    EPSILON = 0.01

    train_x_imp = train_x
    train_y_imp = train_y
    #TODO: impute train
    for f in range(F):
        noise = np.random.rand(S1_train, S2_train)*EPSILON
        train_x_tmp = train_x
        train_x_tmp[:,:,f] += noise
        train_x_imp = np.concatenate((train_x_imp, train_x_tmp), axis=0)
        train_y_imp = np.concatenate((train_y_imp, train_y), axis=0)

    train_x = train_x_imp
    train_y = train_y_imp

    train_dt = np.ones(np.shape(train_y))

    max_time_dist = np.max(train_dt)
    mu = np.mean(train_x)
    std = np.std(train_x)
    mu_y = np.mean(train_y)
    std_y = np.std(train_y)

    train_x = np.clip((train_x-mu)/std, -3, 3)
    train_y = np.clip((train_y-mu_y)/std_y, -3, 3)
    train_dt /= max_time_dist

    vali_x = np.reshape(np.array(x_df[-2*N*T:-N*T]), (N, T, -1))
    vali_y = np.reshape(np.array(y_df[-2*N*T:-N*T]), (N, T, -1))
    S1_vali = np.shape(vali_x)[0]
    S2_vali = np.shape(vali_x)[1]

    vali_x_imp = vali_x
    vali_y_imp = vali_y

    #TODO: impute validation
    for f in range(F):
        noise = np.random.rand(S1_vali, S2_vali)*EPSILON
        vali_x_tmp = vali_x
        vali_x_tmp[:,:,f] += noise
        vali_x_imp = np.concatenate((vali_x_imp, vali_x_tmp), axis=0)
        vali_y_imp = np.concatenate((vali_y_imp, vali_y), axis=0)

    vali_x = vali_x_imp
    vali_y = vali_y_imp

    vali_dt = np.ones(np.shape(vali_y))

    vali_x = np.clip((vali_x-mu)/std, -3, 3)
    vali_y = np.clip((vali_y-mu_y)/std_y, -3, 3)
    vali_dt /= max_time_dist

    test_x = np.reshape(np.array(x_df[-N*T:]), (N, T, -1))
    test_y = np.reshape(np.array(y_df[-N*T:]), (N, T, -1))
    test_dt = np.ones(np.shape(test_y))

    test_x = np.clip((test_x-mu)/std, -3, 3)
    test_y = np.clip((test_y-mu_y)/std_y, -3, 3)
    test_dt /= max_time_dist


    #############################################################################

    BEST_VALI_LOSS = np.nan


    train_num_batches = np.shape(train_x)[0]/N
    vali_num_batches = np.shape(vali_x)[0]/N
    test_num_batches = np.shape(test_x)[0]/N

    train_len = np.shape(train_x)[0]
    vali_len = np.shape(vali_x)[0]
    test_len = np.shape(test_x)[0]



    x_tf = tf.placeholder(tf.float32, shape=(N, T, F))
    dt_tf = tf.placeholder(tf.float32, shape=(N, T, 1))
    y_tf = tf.placeholder(tf.float32, shape=(N, T, 1))


    lstm = LSTMBaseline(hdim)
    prediction = tf.squeeze(lstm.network(x_tf, dt_tf, y_tf, lstm.cell, N, T))


    saver = tf.train.Saver()


    # Define loss and optimizer
    cost = tf.sqrt(tf.reduce_sum((tf.squeeze(prediction) - tf.cast(y_tf, tf.float32))**2)/(N*T))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    # Initializing the variables
    init = tf.global_variables_initializer()


    train_batch_inds = range(train_num_batches)
    vali_batch_inds = range(vali_num_batches)
    test_batch_inds = range(test_num_batches)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(niters):

            #TRAIN
            random.shuffle(train_batch_inds)
            train_loss = 0
            for j in train_batch_inds:
                batch_x = train_x[j*N:(j+1)*N, ...]
                batch_y = train_y[j*N:(j+1)*N, ...]
                batch_dt = train_dt[j*N:(j+1)*N, ...]

                sess.run(optimizer, feed_dict={x_tf: batch_x, y_tf: batch_y, dt_tf: batch_dt})
                loss = sess.run(cost, feed_dict={x_tf: batch_x, y_tf: batch_y, dt_tf: batch_dt})
                print "Minibatch Loss= " + "{:.6f}".format(loss)
                train_loss += loss
            train_loss = train_loss/train_num_batches

            train_file = open(dir_path+'loss_train.txt', 'a+')
            train_file.write("Train Total Loss= " + "{:.5f}".format(train_loss) + "\n")
            train_file.close()


            #VALIDATON
            vali_loss=0
            for j in vali_batch_inds:
                batch_x = vali_x[j*N:(j+1)*N, ...]
                batch_y = vali_y[j*N:(j+1)*N, ...]
                batch_dt = vali_dt[j*N:(j+1)*N, ...]

                loss = sess.run(cost, feed_dict={x_tf: batch_x, y_tf: batch_y, dt_tf: batch_dt})
                predicted_HAMD = sess.run(prediction, feed_dict={x_tf: batch_x, y_tf: batch_y, dt_tf: batch_dt})
                print "Validation Minibatch Loss= " + "{:.6f}".format(loss)
                vali_loss += loss

                plt.figure(figsize=(16, 4))
                plt.title("Validation Minibatch Loss= " + "{:.6f}".format(loss))
                plt.scatter(range(len(batch_y.flatten())), batch_y.flatten()*std_y+mu_y, label='HDRS', color='green', alpha=0.5)
                plt.scatter(range(len(batch_y.flatten())), predicted_HAMD.flatten()*std_y+mu_y, label='predicted', color='red', alpha=0.5)
                plt.savefig(figs_path+'validation'+str(i)+'.png')

            vali_loss = vali_loss/vali_num_batches
            print "Validation Total Loss= " + "{:.5f}".format(vali_loss)

            validation_file = open(dir_path+'loss_valid.txt', 'a+')
            validation_file.write("Validation Total Loss= " + "{:.5f}".format(vali_loss) + "\n")
            validation_file.close()

            if np.isnan(BEST_VALI_LOSS) or vali_loss < BEST_VALI_LOSS:
                BEST_VALI_LOSS = vali_loss
                saver.save(sess, dir_path+'Model')

        print "Optimization finished!"

        #TEST
        new_saver = tf.train.import_meta_graph(dir_path+'Model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(dir_path))


        test_loss = 0

        for j in test_batch_inds:
            batch_x = test_x[j*N:(j+1)*N, ...]
            batch_y = test_y[j*N:(j+1)*N, ...]
            batch_dt = test_dt[j*N:(j+1)*N, ...]

            loss = sess.run(cost, feed_dict={x_tf: batch_x, y_tf: batch_y, dt_tf: batch_dt})
            predicted_HAMD = sess.run(prediction, feed_dict={x_tf: batch_x, y_tf: batch_y, dt_tf: batch_dt})
            print "Test Minibatch Loss= " + "{:.6f}".format(loss)
            test_loss += loss

            plt.figure(figsize=(16, 4))
            plt.title("Test Minibatch Loss= " + "{:.6f}".format(loss))
            plt.scatter(range(len(batch_y.flatten())), batch_y.flatten()*std_y+mu_y, label='HDRS', color='green', alpha=0.5)
            plt.scatter(range(len(batch_y.flatten())), predicted_HAMD.flatten()*std_y+mu_y, label='predicted', color='red', alpha=0.5)
            plt.savefig(figs_path+'test.png')

        test_file = open(dir_path+'acc_test.txt', 'a+')
        test_file.write("Test Loss= " + "{:.5f}".format(test_loss) + "\n")
        test_file.close()


def run_prediction(HAMD_file):
    MODEL_FILE_NAME = MODEL_FILE[0:-4] + '_' +HAMD_file[0:-4]+'.txt'
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


    predict(all_x, y, 'all')
    predict(pca_x, y, 'PCA')
    predict(kernel_pca_x, y, 'KernelPCA')
    predict(truncated_svd_x, y, 'TruncatedSVD')
    predict(sub_x, y, 'sub_data')
    predict(pca_sub_x, y, 'PCA_sub')
    predict(kernel_pca_sub_x, y, 'KernelPCA_sub')
    predict(truncated_svd_sub_x, y, 'TruncatedSVD_sub')

    # plot_prediction(BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL, BEST_VALIDATION_RMSE)


HAMD_files = ['HAMD_imputed_survey.csv']
# HAMD_files = ['HAMD_original.csv',
#               'HAMD_imputed_linear.csv',
#               'HAMD_imputed_survey.csv']

for HAMD_file in HAMD_files:
    BEST_VALIDATION_RMSE = 1000
    BEST_X = None
    BEST_Y = None
    BEST_TTL = None
    BEST_MDL_NAME = None
    BEST_MDL = None
    run_prediction(HAMD_file)