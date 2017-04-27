import numpy as np
from dimensionality_reduction import *

all_df, x_df, y_df = preprocess_survey_x_y()
x_df_nonan = x_df.fillna(0)
reduced_x_df = reduce_dimensionality(x_df_nonan, pca_n=2, kernel_pca_n=2, truncated_svd_n=2)

y = np.array(y_df['HAMD'])
x1 = np.array(reduced_x_df[['PCA_0', 'PCA_1']])
x2 = np.array(reduced_x_df[['KernelPCA_0', 'KernelPCA_1']])
x3 = np.array(reduced_x_df[['TruncatedSVD_0', 'TruncatedSVD_1']])


print np.shape(y)
plot_reduced_feature(x1, 'PCA')
plot_reduced_feature(x2, 'Kernel PCA')
plot_reduced_feature(x3, 'Truncated SVD')