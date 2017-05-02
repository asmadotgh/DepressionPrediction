
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from my_constants import *


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


def main_function():

    EXP = 'LSTMRegression'
    SEED = 1
    learning_rate = 0.001
    hdim = 100   # H
    DATASET = 'MIMIC_BP'#'MIMIC_glucose'
    N = 1           # mini batch size
    T = 50          # time series length
    niters = 500


    #######################   DATA for MIMIC BP random 5 points missing       ##############
    #######################   DATA for MIMIC glucose random 10 points missing ##############

    HAMD_file = 'HAMD_imputed_survey.csv'
    dir_path = 'LSTM/'

    all_df = pd.read_csv(data_dir+HAMD_file)
    feature_df = pd.read_csv(feature_dir+'daily_all.csv')
    all_df = all_df.merge(feature_df, on=['ID', 'date'], how='outer')
    all_df = all_df.dropna(subset=['HAMD'])
    all_df = convert_one_hot_str(all_df, 'ID')


    y_df = all_df[['HAMD']] #'ID', 'date', 'imputed'
    x_df = all_df[['call_daily_IncomingDismissed_count_call',
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
    # x_df = all_df.drop(['ID','HAMD','date', 'imputed'], inplace=False, axis=1)
    x_df = x_df.fillna(0)

    train_x = np.reshape(np.array(x_df[0:9*N*T]), (9*N, T, -1))
    train_y = np.reshape(np.array(y_df[0:9*N*T]), (9*N, T, -1))
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

    # test_val_place_dt = np.array(pk.load(open('../Data/'+DATASET+'/'+DATASET+'_rnd_test_trueValPlc.pk', 'rb')), dtype=float)[:100, :]
    #val, index of the item before, dt
    # num_missing_vals = np.shape(test_val_place_dt)[1]/3

    # mu_list = [mu, 0, 0] * num_missing_vals
    # std_list = [std, 1, 1] * num_missing_vals
    # max_time_dist_list = [1, 1, max_time_dist] * num_missing_vals
    # test_val_place_dt = (test_val_place_dt - mu_list) / std_list
    # test_val_place_dt /= max_time_dist_list

    #############################################################################

    BEST_VALI_LOSS = np.nan

    F = np.shape(train_x)[2]
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

    # Evaluate model
    #correct_pred = prediction - tf.cast(y_tf, tf.float)
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
                plt.savefig('LSTM/figs/validation'+str(i)+'.png')

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
            plt.savefig('LSTM/figs/test.png')

        test_file = open(dir_path+'acc_test.txt', 'a+')
        test_file.write("Test Loss= " + "{:.5f}".format(test_loss) + "\n")
        test_file.close()


main_function()

