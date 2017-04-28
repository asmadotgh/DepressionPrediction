import numpy as np
from dimensionality_reduction import *

all_df, x_df, y_df = preprocess_survey_x_y()
x_df_nonan = x_df.fillna(0)
reduced_x_df, reduced_n = reduce_dimensionality(x_df_nonan, max_n=50, threshold=EXPLAINED_VARIANCE_THRESHOLD)

y = np.array(y_df['HAMD'])
x1 = np.array(reduced_x_df[['PCA_'+str(i) for i in range(reduced_n)]])
x2 = np.array(reduced_x_df[['KernelPCA_'+str(i) for i in range(reduced_n)]])
x3 = np.array(reduced_x_df[['TruncatedSVD_'+str(i) for i in range(reduced_n)]])


print np.shape(y)
plot_reduced_feature(x1, 'PCA')
plot_reduced_feature(x2, 'Kernel PCA')
plot_reduced_feature(x3, 'Truncated SVD')