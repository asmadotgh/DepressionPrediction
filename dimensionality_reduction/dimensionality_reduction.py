import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
import pandas as pd
from my_constants import *

HAMD = pd.read_csv(data_dir+'daily_survey_HAMD.csv')

df = HAMD[HAMD['group']=='MDD']
df = df.fillna(0)
df = df.reset_index(drop = True)
y_df = df[['ID', 'HAMD']]
x_df = df.drop(['date','Name','PSS','group',
                'morning_trigger_time','beverages_trigger_time','medication_trigger_time',
                'feeling1_trigger_time','feeling2_trigger_time','evening_trigger_time',
               'morning_missing', 'beverages_missing', 'medication_missing',
               'feeling1_missing','feeling2_missing','evening_missing',
               'sleepMed', 'anxietyMed', 'painMed' ,'otherMed',
               'beerT', 'wineT' ,'spiritsT', 'ciderT', 'coffeeT', 'teaT', 'sodaT', 'energyT',
               'sleepLog', 'ID'], inplace=False, axis=1)

y = np.array(y_df['HAMD'])
x = np.array(x_df)

def plot_PCA(x, ttl):
    print np.shape(x)
    print np.min(x[:,0])
    print np.max(x[:,0])
    print np.min(x[:,1])
    print np.max(x[:,1])
    #from mpl_toolkits.mplot3d import Axes3D
    plt.figure(figsize=(16,4))
    #ax = fig.add_subplot(111, projection='3d')
    plt.scatter(x[:,0], x[:,1], alpha=0.8, s=20)
    #ax.scatter(x[:,0], x[:,1], x[:,2], label=user, alpha=0.8, s=10)
    # plt.scatter(x, y, label=user, alpha=0.8, s=100)
    # plt.ylim(0,40)
    plt.title(ttl)
    plt.show()

pca = PCA(n_components=2)
x1 = pca.fit_transform(x)

kernel_pca = KernelPCA(n_components=2, kernel='rbf')
x2 = kernel_pca.fit_transform(x)

truncated_svd = TruncatedSVD(n_components=2)
x3 = truncated_svd.fit_transform(x)


print np.shape(y)
plot_PCA(x1, 'PCA')
plot_PCA(x2, 'Kernel PCA')
plot_PCA(x3, 'Truncated SVD')