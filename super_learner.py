import IScore.call_iscore as c_iscore
import logger
import classifier
from sklearn.metrics import mean_squared_error
import pandas
import numpy as np
from sklearn import model_selection
import math
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer


def evaluation_error(learner, test_X_data, test_y_data):
    #classifier = learn(train_df, target_feature_name, temp_feature_subset, learner_name, neighbors_num)
    pred_y_data = learner.predict(test_X_data)
    return mean_squared_error(test_y_data, pred_y_data)
    #return mean_squared_error(learner, test_X_data, test_y_data)


def get_learner_instance(learner_name, neighbors_num):

    # Some input parameters are hard coded, you might need to change it
    if learner_name == classifier.ClassifierType.LINEAR_REGRESSION:
        return classifier.Linear_regression()  # Create linear regression object

    elif learner_name == classifier.ClassifierType.SVR_RBF:
        return classifier.SVR_RBF()

    elif learner_name == classifier.ClassifierType.SVR_LINEAR:
        return classifier.SVR_linear()

    elif learner_name == classifier.ClassifierType.SVR_POLYNOMIAL:
        return classifier.SVR_polynomial()

    elif learner_name == classifier.ClassifierType.KERNEL_RIDGE:
        return classifier.Kernel_ridge()

    elif learner_name == classifier.ClassifierType.KNEIGHBORS_REGRESSOR:
        return classifier.KNeighbors_regressor(neighbors_num)

    elif learner_name == classifier.ClassifierType.DECISION_TREE_REGRESSOR:
        return classifier.Decision_tree_regressor()

    else:
        print('Error: The requested machine learning algorithm is not defined!')
        print('Requested learner name: ', learner_name)
        return None


def learn(X_dependent_data, y_indep_data, learner_name, neighbors_num):
    learner = get_learner_instance(learner_name, neighbors_num)
    # Train the model using the training sets
    # fit() deos internal to_float() conversion. So, please change all the non-degit strings in data to
    # a corresponding digit
    learner.fit(X_dependent_data, y_indep_data)
    return learner


def iscore_handler(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval):
    max_score_subsets = c_iscore.feature_selection(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval)  # TODO check if the feature selection excludes the target column from the dependant column
    return max_score_subsets    

def get_dependent_data(df, feature_set):
    cols = [i.replace('dummy', '') for i in feature_set]
    X_dependent_data = df[cols]
    return X_dependent_data


def get_independent_data(df, target_feature_name):
    target_col_name = target_feature_name.replace('dummy', '')
    y_indep_data = df[target_col_name]
    return y_indep_data


def partition(df, num_partition):
    permuted_indices = np.random.permutation(len(df))

    dfs = []
    for i in xrange(num_partition):
        dfs.append(df.iloc[permuted_indices[i::num_partition]])

    return dfs


def mean_squared_error_scorer():
    return make_scorer(mean_squared_error)


def feature_cross_validation(learner, X_data, y_data, k_fold):
    # K-FOLD CROSS VALIDATION
    scoring_metrics = ['neg_mean_squared_error']
    result = model_selection.cross_validate(learner, X_data, y_data, cv=k_fold, scoring=scoring_metrics)
    #result = model_selection.cross_validate(learner, X_data, y_data, cv=k_fold, scoring=mean_squared_error_scorer)
    return result


def find_best_futures_and_learner(train_df, test_df, max_score_subsets, target_feature_name, learner_name, neighbors_num):
    best_feature_set = []
    best_learner = None
    min_error = float("inf")  # Assume errors are positive, otherwise we consider the absolute value

    y_indep_data = get_independent_data(train_df, target_feature_name)
    test_y_data = get_independent_data(test_df, target_feature_name)

    for feature_set in max_score_subsets:
        X_dependent_data = get_dependent_data(train_df, feature_set)
        test_X_data = get_dependent_data(test_df, feature_set)

        learner = learn(X_dependent_data, y_indep_data, learner_name, neighbors_num)
        error = evaluation_error(learner, test_X_data, test_y_data)
        if abs(error) < min_error:
            min_error = abs(error)
            best_learner = learner
            best_feature_set = feature_set
    return best_feature_set, best_learner, min_error


def find_best_features(df, target_feature_name, learner_name, initial_subset_len, bins_num,
                                       iscore_confidence_interval, neighbors_num):
    max_score_subsets = iscore_handler(df, target_feature_name, initial_subset_len, bins_num,
                                       iscore_confidence_interval)
    # K-FOLD CROSS VALIDATION to select the best feature among all candidate features
    y_indep_data = get_independent_data(df, target_feature_name)

    min_error = float('Inf')
    min_subset = []
    for subset in max_score_subsets:
        X_dependent_data = get_dependent_data(df, subset)

        learner = get_learner_instance(learner_name, neighbors_num)
        result = feature_cross_validation(learner, X_dependent_data, y_indep_data, k_fold)

        # Find the model with the minimum error
        test_errors = result['test_neg_mean_squared_error']
        #logger.log('test_errors: ' + str(test_errors))
        #avg_error = sum([abs(e) for e in test_errors]) / float(len(test_errors))## Changed
        avg_error = sum(test_errors) / float(len(test_errors))
        #if avg_error < abs(min_error):## Changed
        if abs(avg_error) < abs(min_error):
            min_error = avg_error
            min_subset = subset
    #print 'min_error =', min_error, '\nmin_subset =', min_subset

    return min_subset, min_error


def super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, k_fold, neighbors_num):
    # initialization
    partitions = partition(data_frame, k_fold)

    best_learner_name = ''
    best_feature_set = []
    min_error = float('Inf')

    for learner_name in classifier.ClassifierType:
        logger.log("Start: " + str(learner_name))

        avg_error = 0
        tmp_feature_set = []  # HOW CAN WE SAY WHICH tmp_feature_set OF EACH FOLD SHOULD BE THE REPRESENTATIVE OF
        # all k folds?
        best_tmp_error = float('Inf')
        best_tmp_set = []
        logger.log("Doing feature selection based on cross-validation")
        for i in xrange(len(partitions)):
            logger.log("fold no. " + str(i))
            test_df = partitions[i]
            train_df = pandas.concat(partitions[:i] + partitions[i+1:])
            # max_score_subsets = iscore_handler(train_df, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval)

            #tmp_feature_set, tmp_learner, tmp_error = find_best_futures_and_learner(train_df, test_df, max_score_subsets, target_feature_name, learner_name, neighbors_num)
            tmp_feature_set, dummy_error = find_best_features(train_df, target_feature_name, learner_name,
                                                              initial_subset_len, bins_num, iscore_confidence_interval,
                                                              neighbors_num)

            # Learn and estimate based on the selected features (i.e., tmp_feature_set)
            X_dependent_data = get_dependent_data(train_df, tmp_feature_set)
            y_indep_data = get_independent_data(train_df, target_feature_name)
            tmp_learner = learn(X_dependent_data, y_indep_data, learner_name, neighbors_num)

            test_X_data = get_dependent_data(test_df, tmp_feature_set)
            test_y_data = get_independent_data(test_df, target_feature_name)
            # Predict
            pred_y_data = tmp_learner.predict(test_X_data)
            # Calculating error: mean_squared_error
            tmp_error = mean_squared_error(test_y_data, pred_y_data)
            print("temp_error: " + str(tmp_error))
            logger.log("Fold error: " + str(tmp_error))
            logger.log("Fold selected features: " + str(tmp_feature_set))
            avg_error += abs(tmp_error)
            # if tmp_error < best_tmp_error:  # We keep the set with minimum error among all the k-fold
            #     best_tmp_error = tmp_error
            #     best_tmp_set = tmp_feature_set

        avg_error /= len(partitions)
        # logger.log("Selected features for " + str(learner_name) + ": " + str(best_tmp_set))
        logger.log("Average error: " + str(avg_error))
        if avg_error < min_error:
            best_learner_name = learner_name
            # best_feature_set = best_tmp_set
            min_error = avg_error
    logger.log("Best learner of all is: " + str(best_learner_name))
    logger.log("Min error of the best learner: " + str(min_error))

    selected_feature_set, dummy_error = find_best_features(data_frame, target_feature_name, learner_name,
                                                           initial_subset_len, bins_num, iscore_confidence_interval,
                                                           neighbors_num)
    logger.log("Best feature set of all: " + str(selected_feature_set))
    X_dependent_data = get_dependent_data(data_frame, selected_feature_set)
    y_indep_data = get_independent_data(data_frame, target_feature_name)
    best_learner = learn(X_dependent_data, y_indep_data, best_learner_name, neighbors_num)

    return best_learner, selected_feature_set


# External Cross-validation
def SL_cross_validation(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold, neighbors_num):
    logger.log("Doing cross-validation")
    partitions = partition(data_frame, k_fold)
    avg_error = 0
    for i in xrange(len(partitions)):
        test_df = partitions[i]
        train_df = pandas.concat(partitions[:i] + partitions[i+1:])

        tmp_learner, tmp_features = super_learner(train_df, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold, neighbors_num)
        
        test_X_data = get_dependent_data(test_df, tmp_features)
        test_y_data = get_independent_data(test_df, target_feature_name)
        error = evaluation_error(tmp_learner, test_X_data, test_y_data)
        avg_error += error
    
    avg_error /= kfold
    return avg_error


def apply_super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold, neighbors_num):

    learning_model, features = super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold, neighbors_num)
    logger.log("Best learner details: " + str(learning_model))
    error = SL_cross_validation(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, kfold, neighbors_num)
    logger.log("Error of best learner (based on cross-validation): " + str(error))
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
    import datetime
    global logger

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H:%M")
    logger = logger.Logger("log_" + str(date_time) + ".txt")

    # Initialization
    logger.log('Initialization...')
    f_addr = '/home/seyedmah/Desktop/normalized_data_Jan10(Exon_Malueka_Category_C-0_A-1).xlsx'
    target_feature_name = 'skip_percentage'
    initial_subset_len = 55  # can be set to any number, we set it to all number of features
    bins_num = 11  # It is fixed according to convert_normalized_to_discrete function
    iscore_confidence_interval = 0.0001
    k_fold = 5
    skipping_thresh_value = 0.3
    neighbors_num = 5
    thresh_value = 0.3

    # Read input
    logger.log('Read input...')
    data_frame = c_iscore.read_file(f_addr)

    # Learn ML model
    logger.log('Learn ML Model...')
    learning_model, avg_error, feature_subset = apply_super_learner(data_frame, target_feature_name, initial_subset_len, bins_num, iscore_confidence_interval, k_fold, neighbors_num)
    # Predict
    X_data = get_dependent_data(data_frame, feature_subset)
    observed_y_data = get_independent_data(data_frame, target_feature_name)
    predict_y_data = learning_model.predict(X_data)

    # Print Information
    print()
    logger.log('\nMachine learning model:' + str(learning_model))
    print('Machine learning model:', learning_model)
    logger.log('Features: ' + str(feature_subset))
    print('Features: ', feature_subset)  # TODO check if the order of the features are the same as the coefficients
    logger.log('Mean squared error: ' + str(avg_error))
    print('Mean squared error:%.2f' % avg_error)

    logger.log('input features:' + str(X_data))
    print('input features:', X_data)
    logger.log('Observed: ' + str(observed_y_data.as_matrix()))
    print('Observed: ', observed_y_data.as_matrix())
    logger.log('Predicted: ' + str(predict_y_data))
    print('Predicted: ', predict_y_data)
    accuracy = calculate_accuracy(observed_y_data, predict_y_data, thresh_value)
    logger.log('Accuracy: ' + str(accuracy))
    print('Accuracy: ', accuracy)


    # Plot
    logger.log("Plot the diagram...")
    step = 0.01
    x_thresh = np.arange(-0.1, 1, step)
    y_thresh = np.arange(0.0, 1, step)
    x_threshold_line = [thresh_value for i in range(0, len(x_thresh))]
    y_threshold_line = [thresh_value for i in range(0, len(y_thresh))]
    x_axis_len = int(math.ceil(max(predict_y_data)))
    t = np.arange(0.0, x_axis_len, step)

    plt.plot(predict_y_data, observed_y_data, 'go', t, t, 'r-', x_thresh, x_threshold_line, 'b--', y_threshold_line,
             y_thresh, 'b--')
    #    plt.title(testName + ' ($R^2$: ' + "{0:.3f}".format(R2) + ")", fontsize = 14)
    plt.xlabel('Predicted Skip(%)', fontsize=12, weight='bold')
    plt.ylabel('Observed Skip(%)', fontsize=12, weight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.0), fancybox=True, shadow=True, fontsize=10)
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.05, top=0.97, wspace=0.15, hspace=0.3)
    plt.grid(True)
    logger.log("Save the diagram as .pdf and .png")
    plt.savefig('with_learner_accuracy' + str(accuracy) + '.png', bbox_inches='tight')
    plt.savefig('with_learner_accuracy' + str(accuracy) + '.pdf', bbox_inches='tight')
    plt.show()
