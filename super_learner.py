import IScore.call_iscore as c_iscore
import IScore.iscore as iscore
from sklearn import model_selection, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pandas
import numpy as np
import matplotlib.pyplot as plt
import math


def evaluation_error(learner, test_X_data, test_y_data):
    #classifier = learn(train_df, target_feature_name, temp_feature_subset, learner_name)
    pred_y_data = learner.predict(test_X_data)
    return mean_squared_error(test_y_data, pred_y_data)


def learn(X_dependent_data, y_indep_data, learner_name):
#    columns = [i.replace('dummy', '') for i in feature_subset]
#    X_dependent_data = df[columns]
#    y_indep_data = df[target_feature_name.replace('dummy', '')]
    #print(columns)
    learner = get_learner_instance(learner_name)
    # Train the model using the training sets
    learner.fit(X_dependent_data, y_indep_data)
    return learner


def iscore_handler(data_frame, target_feature_name, initial_subset_len, bins_num, error_range):
    # We copy the data_frame. Since the train data combination is different every 
    # round, the discritization will be different and the changes remain in the 
    # data frame (that's why we copy).
    df_cp = data_frame.copy()            

    df = c_iscore.convert_normalized_to_discrete_equal_bin(df_cp, bins_num)  # I-Score works only with descrete values
    max_score_subsets = c_iscore.feature_selection(df, target_feature_name, initial_subset_len, bins_num, error_range) # TODO check if the feature selection excludes the target column from the dependant column
    return max_score_subsets    


def find_best_futures_and_learner(train_df, test_df, max_score_subsets, target_feature_name, learner_name):
    best_feature_set = []
    best_learner = None
    min_error = float("inf")  # Assume errors are positive, otherwise we consider the absolute value

    for feature_set in max_score_subsets:
        cols = [i.replace('dummy', '') for i in feature_set]
        X_dependent_data = train_df[cols]
        target_col_name = target_feature_name.replace('dummy', '')
        y_indep_data = train_df[target_col_name]

        test_X_data = test_df[cols]
        test_y_data = test_df[target_col_name]

        learner = learn(X_dependent_data, y_indep_data, learner_name)
        error = evaluation_error(learner, test_X_data, test_y_data)
        if abs(error) < min_error:
            min_error = abs(error)
            best_learner = learner
            best_feature_set = feature_set
    return best_feature_set, best_learner, min_error
        


def super_learner(data_frame, bin_num, kfold):
    # initialization
    learners_types = get_learner_types()  # TODO define the function
    partitions = partition(df, k_fold)

    best_learner_name = ''

    for learner_name in learner_types:
        min_error = float('Inf')
        avg_error = 0
        best_feature_set = []
        tmp_feature_set = []  # HOW CAN WE SAY WHICH tmp_feature_set OF EACH FOLD SHOULD BE THE REPRESENTATIVE OF
        # all k folds?
        for i in xrange(len(partitions)):
            test_df = partitions[i]
            train_df = pandas.concat(partitions[:i] + partitions[i+1:])
            max_score_subsets = iscore_handler(data_frame, target_feature_name, initial_subset_len, bins_num, error_range)

            tmp_feature_set, tmp_learner, tmp_error = find_best_futures_and_learner(train_df, test_df, max_score_subsets, target_feature_name, learner_name)
                
            avg_error += tmp_error

        avg_error /= len(partitions)
        if avg_error < min_error:
            best_learner_name = learner_name

