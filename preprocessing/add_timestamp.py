import pandas as pd
from datetime import datetime
import datetime as dt
import os
import xml.etree.ElementTree as ET
from my_constants import *

#####################################################################################
################        Add actual timestamp to the datasets        #################
#####################################################################################


DIR = combined_log_dir

def add_timestamp():
    for dname in os.listdir(DIR):
        if '_' in dname or '-' in dname:
            continue
        info_fname=DIR+dname+'/unisens.xml'
        tree=ET.parse(info_fname)
        root = tree.getroot()
        timestamp=root.get('timestampStart')
        timestamp=datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
        def add_date(series):
            series['datetime'] = str(timestamp + dt.timedelta(seconds=series.iloc[0]))
            return series
        for fname in os.listdir(DIR+dname):
            if fname.startswith('.') or fname.startswith('unisens.') or '_' in fname:
                continue
            df = pd.read_csv(DIR+dname+'/'+fname, header=None)
            df = df.apply(add_date, axis=1)
            df.to_csv(DIR+dname+'/'+fname[0:-4]+'_edited'+'.csv',index=False, header=None)


# UNCOMMENT this line to add timestamp to location and create the Loc_edited.csv files
# add_timestamp()
