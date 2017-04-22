HC = ['M001','M002', 'M014', 'M021']
MDD = ['M004', 'M005', 'M006', 'M008', 'M011', 'M012', 'M013', 'M015', 'M016', 'M017', 'M020', 'M022']
outliers = ['M007', 'M018', 'M023'] #dropped out. any user that needs to be removed. for example is in the HAMD scores but doesn't have data or vice versa
NO_OUTGOING_SMS = ['M006', 'M008', 'M011', 'M013', 'M014', 'M015', 'M016', 'M018', 'M020']
NEEDS_CLEANING=['M007', 'M008','M011','M015', 'M020', 'M022']

data_dir = '../data/'
survey_dir = data_dir + 'raw_survey/'
feature_dir = data_dir + 'features/'
raw_log_dir = data_dir + 'raw_logs/'
combined_log_dir = data_dir + 'combined_logs/'

ACC_THRESHOLD = 1000 # radius accuracy in meters for location data
CALL_TYPES = ['Incoming',  'IncomingMissed',  'IncomingDismissed',
              'Outgoing', 'OugtoingNotReached']
SMS_TYPES = ['Incoming', 'Outgoing']

#intervals used for calculating features in different time bins
intervals = [[0, 3, '0_to_3'], [3, 6, '3_to_6'], [6, 9, '6_to_9'], [9, 12, '9_to_12'],
             [12, 15, '12_to_15'], [15, 18, '15_to_18'], [18, 21, '18_to_21'],
             [21, 24, '21_to_24'], [0, 24, 'daily']]

def date_index2str(series):
    series['date'] = series['datetime'].strftime('%Y-%m-%d')
    return series
#sensor_data_dir='sleep_sensor_data/'