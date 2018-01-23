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


def get_learner_instance(learner_name):

    # Some input parameters are hard coded, you might need to change it
    if learner_name == 'linear_regression':
        return linear_model.LinearRegression() # Create linear regression object

    elif learner_name == 'SVR_RBF':
        return SVR(kernel='rbf', C=1e3, gamma=0.1)

    elif learner_name == 'SVR_linear':
        return SVR(kernel='linear', C=1e3)

    elif learner_name == 'SVR_polynomial':
        return SVR(kernel='poly', C=1e3, degree=2)

    elif learner_name == 'kernel_ridge':
        return KernelRidge(alpha=1.0)

    else:
        print('Error: The requested machine learning algorithm is not defined!')
        return None


def learn(X_dependent_data, y_indep_data, learner_name):
#    columns = [i.replace('dummy', '') for i in feature_subset]
#    X_dependent_data = df[columns]
#    y_indep_data = df[target_feature_name.replace('dummy', '')]
    #print(columns)
    learner = get_learner_instance(learner_name)
    # Train the model using the training sets
    learner.fit(X_dependent_data, y_indep_data)
    return learner


def iscore_handler(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval):
    # We copy the data_frame. Since the train data combination is different every 
    # round, the discritization will be different and the changes remain in the 
    # data frame (that's why we copy).
    df_cp = data_frame.copy()            

    df = c_iscore.convert_normalized_to_discrete_equal_bin(df_cp, bins_num)  # I-Score works only with descrete values
    max_score_subsets = c_iscore.feature_selection(df, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval)  # TODO check if the feature selection excludes the target column from the dependant column
    return max_score_subsets    

def get_dependent_data(df, feature_set):
    cols = [i.replace('dummy', '') for i in feature_set]
    X_dependent_data = df[cols]
    return X_dependent_data


def get_independent_data(df, target_feature_name):
    target_col_name = target_feature_name.replace('dummy', '')
    y_indep_data = train_df[target_col_name]
    return y_indep_data


def find_best_futures_and_learner(train_df, test_df, max_score_subsets, target_feature_name, learner_name):
    best_feature_set = []
    best_learner = None
    min_error = float("inf")  # Assume errors are positive, otherwise we consider the absolute value

    for feature_set in max_score_subsets:
#        cols = [i.replace('dummy', '') for i in feature_set]
#        X_dependent_data = train_df[cols]
        X_dependent_data = get_dependent_data(train_df, feature_set)
#        target_col_name = target_feature_name.replace('dummy', '')
#        y_indep_data = train_df[target_col_name]
        y_indep_data = get_independent_data(train_df, target_feature_name)

        #test_X_data = test_df[cols]
        test_X_data = get_dependent_data(test_df, feature_set)
        #test_y_data = test_df[target_col_name]
        test_y_data = get_independent_data(test_df, target_feature_name)

        learner = learn(X_dependent_data, y_indep_data, learner_name)
        error = evaluation_error(learner, test_X_data, test_y_data)
        if abs(error) < min_error:
            min_error = abs(error)
            best_learner = learner
            best_feature_set = feature_set
    return best_feature_set, best_learner, min_error
        


def super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold):
    # initialization
    learners_types = get_learner_types()  # TODO define the function
    partitions = partition(df, k_fold)

    best_learner_name = ''

    best_feature_set = []
    for learner_name in learner_types:
        min_error = float('Inf')
        avg_error = 0
        tmp_feature_set = []  # HOW CAN WE SAY WHICH tmp_feature_set OF EACH FOLD SHOULD BE THE REPRESENTATIVE OF
        # all k folds?
        best_tmp_error = float('Inf')
        best_tmp_set = []
        for i in xrange(len(partitions)):
            test_df = partitions[i]
            train_df = pandas.concat(partitions[:i] + partitions[i+1:])
            max_score_subsets = iscore_handler(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval)

            tmp_feature_set, tmp_learner, tmp_error = find_best_futures_and_learner(train_df, test_df, max_score_subsets, target_feature_name, learner_name)
                
            avg_error += tmp_error
            if tmp_error < best_tmp_error: # We keep the set with minimum error among all the k-fold
                best_tmp_error = tmp_error
                best_tmp_set = tmp_feature_set

        avg_error /= len(partitions)
        if avg_error < min_error:
            best_learner_name = learner_name
            best_feature_set = best_tmp_set
    X_dependent_data = get_dependent_data(data_frame, best_feature_set)
    indep_data = get_independent_data(data_frame, target_feature_name)    
    best_learner = learn(X_dependent_data, y_indep_data, learner_name)       
    
    return best_learner, best_feature_set

# External Cross-validation
def SL_cross_validation(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold):
    partitions = partition(df, k_fold)
    avg_error = 0
    for i in xrange(len(partitions)):
        test_df = partitions[i]
        train_df = pandas.concat(partitions[:i] + partitions[i+1:])
        
        tmp_learner, tmp_features = super_learner(train_df, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold)
        
        test_X_data = get_dependent_data(test_df, feature_set)
        test_y_data = get_independent_data(test_df, target_feature_name)
        error = evaluation_error(learner, test_X_data, test_y_data)
        avg_error += error
    
    avg_error /= kfold
    return avg_error
      
def apply_super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold):
    learning_model, features = super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold)
    error = SL_cross_validation(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold)

    return learning_model, features, error


def calculate_accuracy(observed_list, predict_list, thresh_value):
    assert len(observed_list) == len(predict_list)
    observed_yes_predict_no = []
    observed_no_predict_no = []
    observed_yes_predict_yes = []
    observed_no_predict_yes = []
    for i in range(len(observed_list)):
        if observed_list[i] > thresh_value and predict_list[i] > thresh_value:
            observed_yes_predict_yes.append((observed_list[i], predict_list[i]))
        elif observed_list[i] > thresh_value and predict_list[i] < thresh_value:
            observed_yes_predict_no.append((observed_list[i], predict_list[i]))
        elif observed_list[i] < thresh_value and predict_list[i] > thresh_value:
            observed_no_predict_yes.append((observed_list[i], predict_list[i]))
        elif observed_list[i] < thresh_value and predict_list[i] < thresh_value:
            observed_no_predict_no.append((observed_list[i], predict_list[i]))
    return len(observed_yes_predict_yes + observed_no_predict_no)/float(len(observed_list))


if __name__ == '__main__':
    #Initialization
    f_addr = '/home/seyedmah/Desktop/normalized_data_Jan10.xlsx'
    target_feature_name = 'skip_percentage'
    initial_subset_len = 55 # can be set to any number, we set it to all number of features
    bins_num = 11 # It is fixed according to convert_normalized_to_discrete function
    iscore_confidence_interval = 0.0001
    k_fold = 5
    learner_name = 'SVR_polynomial'
    skipping_thresh_value = 0.3

    #Read input
    df = c_iscore.read_file(f_addr)

    # Learn ML model
    learning_model, avg_error, feature_subset = apply_super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold)   
    # Predict
    X_data = get_dependent_data(df, feature_set)
    observed_y_data = get_independent_data(df, target_feature_name)
    predict_y_data = learning_model.predict(X_data)

    # Print Information
    print()
    print('Machine learning model:', learning_model)
    print('Features: ', feature_subset) # TODO check if the order of the features are the same as the coefficients
#    print('Coefficients: ', learner.coef_)
    print('Intercept:', learner.intercept_)
    print('Mean squared error:%.2f' %avg_error)
    #print(learner.get_params(deep=True)) 

    print('input features:', X_data)
    print('Observed: ', observed_y_data.as_matrix())
    print('Predicted: ', predict_y_data)
    print 'Accuracy: ', calculate_accuracy(observed_y_data, predict_y_data, thresh_value)

    # Plot
    step = 0.01
    x_thresh = np.arange(-0.1, 1, step)
    y_thresh = np.arange(0.0, 1, step)
    x_threshold_line = [thresh_value for i in range(0, len(x_thresh))]
    y_threshold_line = [thresh_value for i in range(0, len(y_thresh))]
    x_axis_len = int(math.ceil(max(predict_y_data)))
    t= np.arange(0.0, x_axis_len, step) 
