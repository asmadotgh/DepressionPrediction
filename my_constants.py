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

#intervals used for calculating features in different time bins
intervals = [[0, 3, '0_to_3'], [3, 6, '3_to_6'], [6, 9, '6_to_9'], [9, 12, '9_to_12'],
             [12, 15, '12_to_15'], [15, 18, '15_to_18'], [18, 21, '18_to_21'],
             [21, 24, '21_to_24'], [0, 24, 'daily']]

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
                    'call_daily_Incoming_median_call_duration',
                    'call_daily_Incoming_std_call_duration',
                    'call_daily_Incoming_sum_call_duration',
                    'call_daily_Outgoing_count_call',
                    'call_daily_Outgoing_mean_call_duration',
                    'call_daily_Outgoing_median_call_duration',
                    'call_daily_Outgoing_std_call_duration',
                    'call_daily_Outgoing_sum_call_duration',
                    'call_daily_incoming_outgoing_call_duration',
                    'call_daily_incoming_outgoing_call_count']
DISPLAY_SUB_FEATURES = ['display_daily_sum_on_duration',
                        'display_daily_std_on_duration',
                        'display_daily_mean_on_duration',
                        'display_daily_median_on_duration',
                        'display_daily_count_on']
LOCATION_SUB_FEATURES = ['location_daily_count',
                        'location_daily_total_std']
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
MOTION_SUB_FEATURES = ['motion_average_motion_24hrs_left',
                       'motion_average_motion_24hrs_right',
                       'motion_fraction_time_in_motion_24hrs_left',
                       'motion_fraction_time_in_motion_24hrs_right',
                       'motion_median_motion_24hrs_left',
                       'motion_median_motion_24hrs_right',
                       'motion_recording_time_fraction_24hrs_left',
                       'motion_recording_time_fraction_24hrs_right',
                       'motion_std_motion_24hrs_left',
                       'motion_std_motion_24hrs_right']

ONE_HOT_USERS = ['ID_M004', 'ID_M006', 'ID_M008', 'ID_M011', 'ID_M012', 'ID_M013',
                        'ID_M015', 'ID_M016', 'ID_M017', 'ID_M020', 'ID_M022']

SUB_FEATURES = CALL_SUB_FEATRURES+DISPLAY_SUB_FEATURES+LOCATION_SUB_FEATURES+\
               SLEEP_SUB_FEATURES+SMS_SUB_FEATURES+MOTION_SUB_FEATURES+ONE_HOT_USERS

