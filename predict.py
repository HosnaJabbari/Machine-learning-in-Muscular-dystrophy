import IScore.call_iscore as c_iscore
import IScore.iscore as iscore
from sklearn import model_selection, linear_model
import pandas


def cross_validation(lreg, X_data, y_data, k_fold):
    # K-FOLD CROSS VALIDATION
    scoring_metrics = ['neg_mean_squared_error']
    results = model_selection.cross_validate(lreg, X_data, y_data, cv=k_fold, scoring=scoring_metrics)





def find_Model(data_frame, target_feature_name, initial_subset_len, k_fold, bins_num, error_range):
    # FEATURE SELECT0ION
    max_subsets = c_iscore.feature_selection(data_frame, target_feature_name, initial_subset_len, bins_num, error_range)

    for i in max_subsets: 
        print len(i), i
    

    #LEARNING ALGORITHM
    # Create linear regression object
    lreg = linear_model.LinearRegression()

    # Train the model using the training sets
#    lreg.fit(X_train, y_train)

    # Make predictions using the testing set
#    diabetes_y_pred = regr.predict(X_test)


    # K-FOLD CROSS VALIDATION
#    scoring_metrics = ['neg_mean_squared_error']
#   results = model_selection.cross_validate(lreg, X_data, y_data, cv=k, scoring=scoring_metrics)
    y_data = data_frame[target_feature_name]
    
    for subset in max_subsets:
        col = [i.replace('dummy', '') for i in subset]
        #X_data = df[list(subset)]
        X_data = data_frame[col]
        print X_data
       
        result = cross_validation(lreg, X_data, y_data, k_fold) 
        print(result)
        return



if __name__ == '__main__':
    
    #Initialization
    f_addr = '/home/seyedmah/Desktop/normalized_data_Dec19.xlsx'
    target_feature_name = 'skip_percentage'
    initial_subset_len = 54
    bins_num = 11#It is fixed according to convert_normalized_to_discrete function
    error_range = 0.0001
    k_fold = 7

    #Read input
    data_frame = c_iscore.read_file(f_addr)
    df = c_iscore.convert_normalized_to_discrete_equal_bin(data_frame, bins_num)
   
    model = find_Model(df, target_feature_name, initial_subset_len, k_fold, bins_num, error_range)
