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


def cross_validation(learner, X_data, y_data, k_fold):
    # K-FOLD CROSS VALIDATION
    scoring_metrics = ['neg_mean_squared_error']
    result = model_selection.cross_validate(learner, X_data, y_data, cv=k_fold, scoring=scoring_metrics)
    return result



# Select best feature subset among the iscore suggested subsets
def find_best_features(data_frame, target_feature_name, initial_subset_len, k_fold, bins_num, error_range, learner_name):
    # FEATURE SELECT0ION
    df = c_iscore.convert_normalized_to_discrete_equal_bin(data_frame, bins_num) # I-Score works only with descrete values
    max_subsets = c_iscore.feature_selection(df, target_feature_name, initial_subset_len, bins_num, error_range) # TODO check if the feature selection excludes the target column from the dependant column

#    for i in max_subsets: 
#        print len(i), i
    
    # Make predictions using the testing set
#    diabetes_y_pred = regr.predict(X_test)


    # K-FOLD CROSS VALIDATION
#    scoring_metrics = ['neg_mean_squared_error']
#   results = model_selection.cross_validate(learner, X_data, y_data, cv=k, scoring=scoring_metrics)
    y_data = df[target_feature_name.replace('dummy', '')]
    
    min_error = float('Inf')
    min_subset = []
    for subset in max_subsets:
        cols = [i.replace('dummy', '') for i in subset]
        #X_data = df[list(subset)]
        X_data = df[cols]
        #print X_data
        
        learner = get_learner_instance(learner_name) 
        #Internal cross-validation   
        result = cross_validation(learner, X_data, y_data, k_fold) 
        #print(result)
         

        # Find the model with the minimum error
        test_errors = result['test_neg_mean_squared_error']
        avg_error = sum(test_errors)/float(len(test_errors)) 
#        print test_errors, avg_error
        if avg_error < abs(min_error):
            min_error = avg_error
            min_subset = subset
    print 'min_error =', min_error, '\nmin_subset =', min_subset
    return min_subset, min_error

    
def partition(df, num_partition):
    permuted_indices = np.random.permutation(len(df))

    dfs = []
    for i in xrange(num_partition):
        dfs.append(df.iloc[permuted_indices[i::num_partition]])
       
    return dfs


def learn(df, target_feature_name, feature_subset, learner_name):
    columns = [i.replace('dummy', '') for i in feature_subset]
    y_data = df[target_feature_name.replace('dummy', '')]
    print(columns)
    X_data = df[columns]
    learner = get_learner_instance(learner_name) 
    # Train the model using the training sets
    learner.fit(X_data, y_data)
    return learner


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


def get_learning_model(data_frame, target_feature_name, initial_subset_len, k_fold, bins_num, error_range, learner_name):
    # Feature selection
#    print('old_df', data_frame)
    df = data_frame.copy() # Make a copy of data frame to keep the original data unchanged (since find_best_features first normalizes the data and then discretizes it.)
    feature_subset, dummy_error = find_best_features(df, target_feature_name, initial_subset_len, k_fold, bins_num, error_range, learner_name) # This calculated error is not valid, should be replaced with average errors of the crosvalidation in the following
#    print('new_df', df)
    # Learning using all data rows as input
    learner = learn(data_frame, target_feature_name, feature_subset, learner_name)

    # Cross-validation
    partitions = partition(df, k_fold)
    error_sum = 0
    for i in xrange(len(partitions)):
        test_df = partitions[i]
        train_df = pandas.concat(partitions[:i] + partitions[i+1:])
        # We copy the data_frame. Since the train data combination is different every 
        # round, the discritization will be different and the changes reamain in the 
        # data frame (that's why we copy).
        train_df_cp = train_df.copy()
        temp_feature_subset, dummy_error = find_best_features(train_df_cp, target_feature_name, initial_subset_len, k_fold, bins_num, error_range, learner_name)
        tmp_learner = learn(train_df, target_feature_name, temp_feature_subset, learner_name)
        columns = [i.replace('dummy', '') for i in temp_feature_subset]
        test_X_data = test_df[columns]
        test_y_data = test_df[target_feature_name.replace('dummy', '')]

        # Predict
        pred_y_data = tmp_learner.predict(test_X_data)
        # Calculating error: mean_squared_error
        error_sum += mean_squared_error(test_y_data, pred_y_data)
 
    
    avg_error = float(error_sum)/len(partitions)

    # The coefficients
#    print('Coefficients: \n', learner.coef_)
    # The mean squared error
#    print("Mean squared error: %.2f" %mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
#    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    return learner, avg_error, [i.replace('dummy', '') for i in feature_subset]


if __name__ == '__main__':
    
    #Initialization
    f_addr = '/home/seyedmah/Desktop/normalized_data_Jan10.xlsx'
    target_feature_name = 'skip_percentage'
    initial_subset_len = 55 # can be set to any number, we set it to all number of features
    bins_num = 11 # It is fixed according to convert_normalized_to_discrete function
    error_range = 0.0001
    k_fold = 5
    learner_name = 'SVR_polynomial'
    thresh_value = 0.3

    #Read input
    df = c_iscore.read_file(f_addr)
    #df = c_iscore.convert_normalized_to_discrete_equal_bin(data_frame, bins_num)

    # Learn ML model
    learner, avg_error, feature_subset = get_learning_model(df, target_feature_name, initial_subset_len, k_fold, bins_num, error_range, learner_name)   
    
    # Predict
    columns = [i.replace('dummy', '') for i in feature_subset]
    X_data = df[columns]
    observed_y_data = df[target_feature_name.replace('dummy', '')]
    predict_y_data = learner.predict(X_data)
    
    # Print Information
    print()
    print('Machine learning model:', learner)
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
    
    plt.plot(predict_y_data, observed_y_data, 'go', t, t, 'r-', x_thresh, x_threshold_line, 'b--', y_threshold_line, y_thresh, 'b--')
#    plt.title(testName + ' ($R^2$: ' + "{0:.3f}".format(R2) + ")", fontsize = 14)
    plt.xlabel('Predicted Skip(%)', fontsize = 12, weight = 'bold')
    plt.ylabel('Observed Skip(%)', fontsize = 12, weight = 'bold')
    plt.legend(loc = 'upper left', bbox_to_anchor = (0, 1.0), fancybox = True, shadow = True, fontsize = 10)
    plt.subplots_adjust(left = 0.2, right = 0.9, bottom = 0.05, top = 0.97, wspace = 0.15, hspace = 0.3) 
    plt.grid(True)
    plt.show()
    
    
