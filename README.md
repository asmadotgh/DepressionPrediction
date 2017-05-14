# Predicting Hamilton Depression Rating Scale (HDRS) from passive sensor data

## Steps
### Preprocessing
Run combine_logs.py and add_timestamp.py under preprocessing directory. This will do the preprocessing to combine all phone features together.
### Preprocessing and Feature generation
Run all the *_features.py (except combine_feature) under feature_generation directory. This will create .csv files for each feature type. Then, run combine_features.py to combine them all and create a total daily dataset.
### Dimensionality reduction of survey features
Run dimensionality_reduction.py under dimensionality_reduction directory. You can test it using test_dim_reduction.py, but that is not necessary.
### Predicting HDRS from a surrogate variable (survey data) and imputing HDRS for increasing the size of the dataset
Run HDRS_imputation_survey.py under HDRS_imputation directory. The other two files under the same directory are for creating baseline csv files. 
### Predicting HDRS from passive data (phone sensors, E4 wearable wristbands)
Run *.py (except the ensemble ones) under HDRS_prediction directory. This creates .csv files for prediction using different regressors. Then, run the ensemble.py for the final customized ensemble method.

Note that all constants can be found in the my_constants.py file so that we can regenerate everything. However, the data folder is not on github.
