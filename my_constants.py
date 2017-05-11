import numpy as np

HC = ['M001', 'M002', 'M014', 'M021']
MDD = ['M004', 'M005', 'M006', 'M008', 'M011', 'M012', 'M013', 'M015', 'M016', 'M017', 'M020', 'M022']
outliers = ['M007', 'M018', 'M023'] #dropped out. any user that needs to be removed. for example is in the HAMD scores but doesn't have data or vice versa
NO_OUTGOING_SMS = ['M006', 'M008', 'M011', 'M013', 'M014', 'M015', 'M016', 'M018', 'M020']
NEEDS_CLEANING = ['M007', 'M008','M011','M015', 'M020', 'M022']

data_dir = '../data/'
survey_dir = data_dir + 'raw_survey/'
feature_dir = data_dir + 'features/'
raw_log_dir = data_dir + 'raw_logs/'
combined_log_dir = data_dir + 'combined_logs/'
results_dir = 'results/'

ACC_THRESHOLD = 1000 # radius accuracy in meters for location data
CALL_TYPES = ['Incoming',  'IncomingMissed',  'IncomingDismissed',
              'Outgoing', 'OugtoingNotReached']
SMS_TYPES = ['Incoming', 'Outgoing']
APP_TYPES = ['game', 'email', 'web', 'calendar', 'communication', 'facebook',
             'maps', 'youtube', 'photo', 'shopping', 'clock']

SEED = 123 #12345

MODEL_FILE = 'model.txt'

K_FOLD_N = 10
TEST_RATIO = 0.1
STATIONARY_SPEED = 0.3


# for dimensionality reduction
EXPLAINED_VARIANCE_THRESHOLD = 0.85

REGULARIZATION_ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]
def get_app_type(app):
    if app in ['air.com.sgn.bookoflife.gp']:
        return 'game'
    if app in ['com.android.email', 'com.yahoo.mobile.client.android.mail']:
        return 'email'
    if app in ['com.android.chrome', 'com.sec.android.app.sbrowser']:
        return 'web'
    if app in ['com.android.calendar']:
        return 'calendar'
    if app in ['com.android.contacts', 'com.android.incallui', 'com.android.mms', 'com.android.phone', 'com.whatsapp']:
        return 'communication'
    if app in ['com.facebook.katana', 'com.facebook.orca']:
        return 'facebook'
    if app in ['com.google.android.apps.maps']:
        return 'maps'
    if app in ['com.google.android.youtube']:
        return 'youtube'
    if app in ['com.sec.android.app.camera', 'com.sec.android.gallery3d', 'com.sec.android.mimage.photoretouching']:
        return 'photo'
    if app in ['com.walmart.android', 'com.target.ui', 'com.macys.android']:
        return 'shopping'
    if app in ['com.sec.android.app.clockpackage']:
        return 'clock'


#game, email, web, calendar, communication, facebook, maps, video streaming, photo, shopping, clock

#intervals used for calculating features in different time bins
intervals = [[0, 3, '0_to_3'], [3, 6, '3_to_6'], [6, 9, '6_to_9'], [9, 12, '9_to_12'],
             [12, 15, '12_to_15'], [15, 18, '15_to_18'], [18, 21, '18_to_21'],
             [21, 24, '21_to_24'], [9, 18, 'day_hours'], [0, 24, 'daily']]

def date_index2str(series):
    series['date'] = series['datetime'].strftime('%Y-%m-%d')
    return series
#sensor_data_dir='sleep_sensor_data/'

def is_imputed(series):
    if np.isnan(series['HAMD']):
        series['imputed'] = 'y'
    else:
        series['imputed'] = 'n'
    return series


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


CALL_SUB_FEATRURES = ['call_daily_IncomingDismissed_count_call',
                    'call_daily_IncomingMissed_count_call',
                    'call_daily_Incoming_count_call',
                    'call_daily_Incoming_mean_call_duration',
                    'call_daily_Incoming_std_call_duration',
                    'call_daily_Outgoing_count_call',
                    'call_daily_Outgoing_mean_call_duration',
                    'call_daily_Outgoing_std_call_duration',
                    'call_daily_incoming_outgoing_call_duration',
                    'call_daily_incoming_outgoing_call_count']
DISPLAY_SUB_FEATURES = ['display_daily_sum_on_duration',
                        'display_daily_std_on_duration',
                        'display_daily_mean_on_duration',
                        'display_daily_median_on_duration',
                        'display_daily_count_on']
LOCATION_SUB_FEATURES = ['location_day_hours_total_std']
SLEEP_SUB_FEATURES = ['sleep_24hrs_fraction_recording',
                        'sleep_24hrs_sleep_(s)',
                        'sleep_night_sleep_(s)',
                        'sleep_night_fraction_recording',
                        'sleep_night_sleep_onset_timeelapsed_since_noon_(s)',
                        'sleep_night_max_uninterrupted_sleep_(s)',
                        'sleep_night_nbwakeups',
                        'sleep_ day_wakeup_onset_timeelapsed_since_midnight_(s)',
                        'sleep_sleep_reg_index']
SMS_SUB_FEATURES = ['sms_daily_Incoming_count_sms']
MOTION_SUB_FEATURES = ['motion_average_motion_24hrs',
                       'motion_fraction_time_in_motion_24hrs',
                       'motion_median_motion_24hrs',
                       'motion_recording_time_fraction_24hrs',
                       'motion_std_motion_24hrs']

EDA_SUB_FEATURES = ['eda_eda_24hrs_mean_difference_r_l',
                    'eda_eda_24hrs_mean_left',
                    'eda_eda_24hrs_mean_right',
                    'eda_eda_24hrs_scl_cvx_mean_diff_r_l',
                    'eda_eda_24hrs_scl_cvx_mean_left',
                    'eda_eda_24hrs_scl_cvx_mean_right',
                    'eda_eda_24hrs_scr_cvx_mean_diff_r_l',
                    'eda_eda_24hrs_scr_cvx_mean_left',
                    'eda_eda_24hrs_scr_cvx_mean_right',
                    'eda_eda_24hrs_scrs_diff_r_l',
                    'eda_eda_24hrs_scrs_left',
                    'eda_eda_24hrs_scrs_mean_ampl_diff_r_l',
                    'eda_eda_24hrs_scrs_mean_ampl_left',
                    'eda_eda_24hrs_scrs_mean_ampl_right',
                    'eda_eda_24hrs_scrs_right',
                    'eda_eda_night_mean_difference_r_l',
                    'eda_eda_night_mean_left',
                    'eda_eda_night_mean_right',
                    'eda_eda_night_scl_cvx_mean_diff_r_l',
                    'eda_eda_night_scl_cvx_mean_left',
                    'eda_eda_night_scl_cvx_mean_right',
                    'eda_eda_night_scr_cvx_mean_diff_r_l',
                    'eda_eda_night_scr_cvx_mean_left',
                    'eda_eda_night_scr_cvx_mean_right',
                    'eda_eda_night_scrs_diff_r_l',
                    'eda_eda_night_scrs_left',
                    'eda_eda_night_scrs_mean_ampl_diff_r_l',
                    'eda_eda_night_scrs_mean_ampl_left',
                    'eda_eda_night_scrs_mean_ampl_right',
                    'eda_eda_night_scrs_right']

ONE_HOT_USERS = ['ID_M004', 'ID_M006', 'ID_M008', 'ID_M011', 'ID_M012', 'ID_M013',
                 'ID_M015', 'ID_M016', 'ID_M017', 'ID_M020', 'ID_M022']

SUB_FEATURES = CALL_SUB_FEATRURES+DISPLAY_SUB_FEATURES+LOCATION_SUB_FEATURES+\
               SLEEP_SUB_FEATURES+SMS_SUB_FEATURES+MOTION_SUB_FEATURES+EDA_SUB_FEATURES+\
               ONE_HOT_USERS

IND_TRAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 451, 452, 453, 454, 455, 456, 457, 460, 461, 462, 463, 464, 465, 466, 467, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 527, 528, 529, 530, 531, 532, 533, 534, 535, 537, 538, 539, 540, 541, 542, 543, 544]
IND_TEST = [105, 468, 219, 189, 422, 0, 319, 512, 276, 164, 302, 526, 335, 203, 106, 204, 78, 352, 125, 458, 133, 150, 407, 510, 245, 366, 41, 95, 259, 233, 436, 450, 178, 149, 351, 54, 289, 107, 545, 29, 318, 320, 379, 393, 260, 408, 536, 459, 55, 65, 511, 14, 482, 496]

def split_data_ind(inds, test_N):
    np.random.shuffle(inds)
    ind_train = inds[test_N:]
    ind_test = inds[0:test_N]
    return IND_TRAIN, IND_TEST
    # return ind_train, ind_test

MAX_PCA_ALL = 25
MAX_PCA_SUB = 20
MAX_PCA_SUB_HIST = 25