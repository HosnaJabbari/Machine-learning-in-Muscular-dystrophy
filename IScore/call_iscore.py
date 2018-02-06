import iscore as isc
import pandas
import math
import itertools
import numpy
import string
import equal_bin_discretization as discrete
import sys
import traceback

#The values of each column should be between 0 and 1
#def scale_up(in_list, up_value):
    


def read_file(f_addr):
    df = pandas.read_excel(f_addr)
#    columns = df.columns
    return df


##########################Test if df has changed at the end (i.e., columns is not a view only)
def convert_nominal_to_int(df):  # TODO Should be modified to change the elemenst like: convert_normalized_to_discrete_equal_bin() function
    columns = df.columns.values
    index_list = df.index.values
    for c in columns:
        temp_list = []
        temp_dict = {}
        count = 1
        if type(df[c].values[0]) == type(u''):
            for val in df[c].values:
                if val not in temp_dict:
                    temp_dict[val] = count
                    count += 1
                temp_list.append(temp_dict[val])

            for i in xrange(len(index_list)):
                df.at[index_list[i], c] = temp_list[i]
            #df[c] = pandas.DataFrame({c: temp_list})
    return df


def convert_normalized_to_discrete_equal_bin(df, bins_num):
#    df = data_frame.copy()
    columns = df.columns.values
    index_list = df.index.values
    for col in columns:
        #print(df[c])
        temp_columns = []

        if type(df[col].values[0]) == type(numpy.float64(1.4)):#1.4 is just a random number
            # print "Original: ", df[col]
            for val in df[col].values:
                new_val = int(round(val * 100))
                temp_columns.append(new_val)

            discrete_col, cutoff = discrete.discretize(temp_columns, bins_num)
            for i in xrange(len(index_list)):
                df.at[index_list[i], col] = discrete_col[i]

            # if debug:
            #     print "After discretize: len:", len(discrete_col), discrete_col
            #     print "data frame index: ", df.index.values
            #
            # ##df[c] = pandas.DataFrame({c: discrete_col})
            # if debug:
            #     print "After assignment: len: ", len(df[col]), df[col]
    return df


##########################Test if df has changed at the end (i.e., columns is not a view only)
#Gives us the ganularity of 11
def convert_normalized_to_discrete_equal_section(data_frame):  # TODO Should be modified to change the elemenst like: convert_normalized_to_discrete_equal_bin() function
    df = data_frame.copy()
    columns = df.columns
    for c in columns:
        temp_list = []
        temp_dict = {}
        if type(df[c].values[0]) == type(numpy.float64(1.4)):#1.4 is just a random number
            for val in df[c].values:
                if val not in temp_dict:
                    temp_dict[val] = int(round(val * 10))
                temp_list.append(temp_dict[val])
            df[c] = pandas.DataFrame({c:temp_list})
    return df


def get_all_initial_subsets(columns_label_list, subset_len):
    #return [columns_label_list]
    return set(itertools.combinations(columns_label_list, subset_len))    



#Not very efficient in terms of space
#def initialize_cells(subset_len, granularity_num):
#    #cells = [[] for i in xrange(pow(granularity_num, subset_len))]
#    cells = []
#    count = pow(granularity_num, subset_len)
#    while count > 0:
#        cells.append([])
#        count -= 1
#    return cells



def correct_name(name):
    #No name starts with number or special character, or has : in the name
    invalid_chars = set(string.punctuation)
#    invalid_chars = set(string.punctuation.replace("_", ""))
#    if any(char in invalid_chars for char in word):
    for char in name:
        if char in invalid_chars:
            name = name.replace(char, "_")
    if (name[0] in invalid_chars) or name[0].isdigit():
        name = 'dummy' + name
    return name



def partition(df, features_subset, granularity_num):
    #cells = initialize_cells(len(features_subset), granularity_num)
    cells = {}
#    cell_inx = 0.0
    for row_num, row in enumerate(df.itertuples(), 1):
        height = len(features_subset) - 1
        cell_inx = numpy.longdouble(0)
        for f in features_subset:
#            print 'row: ', getattr(row, f)
#            print 'pow: ', pow(granularity_num, height)
#            print 'before, cell_inx', cell_inx
#            print 'f: ', f
            cell_inx += numpy.longdouble(getattr(row, f)) * numpy.longdouble(pow(granularity_num, height))
#            print 'after, cell_inx:', cell_inx
            height -= 1
        #cells[cell_inx].append(row)
        if cell_inx not in cells:
            cells[cell_inx] = [row]
        else:
            cells[cell_inx].append(row)
    return cells



def get_iscore(df, feature_sample, granularity_num, target_feature_name):
    cells = partition(df, feature_sample, granularity_num)
    target_values = df[target_feature_name].values
    cells_avg = {}

    for key in cells:
        cl = cells[key]
        temp = 0
        #temp_counter = 0
        avg = 0
        for elem in cl:
            temp += getattr(elem, target_feature_name)
        cell_len = len(cl)
        if cell_len != 0:
            avg = float(temp)/len(cl)
        cells_avg[key] = (avg, cell_len)

    return isc.compute_iscore(target_values, cells_avg)


def static_vars(**kwargs):
    """Defines static variable in a function. Currently, there is no built-in implementation in  Python"""

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

# elems_in_range is a list of tuples: (tuple's corresponding value, tuple)
@static_vars(lower_bound=-float("inf"), elems_in_range=[])
def keep_max_elements_with_range(new_element, value, error_range):  # Warning: this method has static variables
    #lower_bound = -float("inf")  # plus inf
    #upper_bound = float("inf")  # minus inf
    old_lower_bound = keep_max_elements_with_range.lower_bound

    if value >= keep_max_elements_with_range.lower_bound:
        # Updating lower and upper bound and add new element
        if value - error_range > keep_max_elements_with_range.lower_bound:
            keep_max_elements_with_range.lower_bound = value - error_range

        # Add new element
        keep_max_elements_with_range.elems_in_range.append((value, new_element))

        # Remove unwanted values based on the new lower bound
        if old_lower_bound != keep_max_elements_with_range.lower_bound:
            keep_max_elements_with_range.elems_in_range.sort(key=lambda tup: tup[0])  # sorts in place based on the value
            for indx in xrange(len(keep_max_elements_with_range.elems_in_range)):
                tmp_value, arr = keep_max_elements_with_range.elems_in_range[indx]
                if tmp_value >= keep_max_elements_with_range.lower_bound:
                    break  # Since the array is sorted the rest are greater than the lower bound

            del keep_max_elements_with_range.elems_in_range[:indx]
    median_value = keep_max_elements_with_range.lower_bound + error_range
    return keep_max_elements_with_range.elems_in_range, median_value


# elems_in_range_container is a list of tuples: (tuple's corresponding value, tuple)
def pick_max_elements_within_range(new_element, value, error_range, lower_bound=-float("inf"), elems_in_range_container=[]):
    #lower_bound = -float("inf")  # plus inf
    #upper_bound = float("inf")  # minus inf
    old_lower_bound = lower_bound

    if value >= lower_bound:
        # Updating lower and upper bound and add new element
        if value - error_range > lower_bound:
            lower_bound = value - error_range

        # Add new element
            elems_in_range_container.append((value, new_element))

        # Remove unwanted values based on the new lower bound
        if old_lower_bound != lower_bound:
            elems_in_range_container.sort(key=lambda tup: tup[0])  # sorts in place based on the value
            for indx in xrange(len(elems_in_range_container)):
                tmp_value, arr = elems_in_range_container[indx]
                if tmp_value >= lower_bound:
                    break  # Since the array is sorted the rest are greater than the lower bound

            del elems_in_range_container[:indx]

    return elems_in_range_container, lower_bound


def sample_uniformly(samples_list, num):
    # return [samples_list[-1]]
    length = len(samples_list)
    # step = length//num
    step = num

    # if step == 0:
    if length <= step:
        return samples_list

    selected_samples = []
    for i in xrange(0, length, step):
        selected_samples.append(samples_list[i])

    return selected_samples


def is_exist(my_dict, current_element):
    key = len(current_element)
    return (key in my_dict and current_element in my_dict[key])


# Backward Dropping Algorithm
def BDA(df, initial_features_sample, granularity_num, target_feature_name, error_range):
    #global_max_iscore = -float('Inf')
    #global_max_subset = []

    #global_max_iscore = get_iscore(df, initial_features_sample, granularity_num, target_feature_name)

    # global_max_subset = [initial_features_sample]
    global_max_subset = []
    global_max_iscore = 0
    all_sample_stars = [(global_max_iscore, initial_features_sample)]

#    print "\n---------------------------------------------- BDA started --------------------------------------------\n"

    while len(all_sample_stars) > 0 and len(all_sample_stars[0][1]) > 1:
        local_max_iscore = -float('Inf')
        local_max_subset = []
        temp_max_iscore = 0
        tmp_max_set = []
        last_candidates = []
        global_dict = {}

        for sample_star in all_sample_stars:
#        print "\nSTART LOCAL #####################################################"
            local_max_iscore = 0

            tmp_lower_bound = -float("inf")
            tmp_elems_in_range_container = []
            count_child = 0
            for i in range(len(sample_star[1])):
    ##            print type([sample_star])
    ##            print 'sample_star: ', sample_star
    ##            print 'first: ', sample_star[:i]
    ##            print 'second: ', sample_star[i+1:]
    ##            print list(sample_star[:i]) + list(sample_star[i+1:])
                local_sample = list(sample_star[1][:i]) + list(sample_star[1][i+1:])
                #Compute I-Score
                if is_exist(global_dict, local_sample):
                    continue
                iscore = get_iscore(df, local_sample, granularity_num, target_feature_name)



                #tmp_elems_in_range_container, tmp_lower_bound = pick_max_elements_within_range(local_sample, iscore, error_range, tmp_lower_bound, tmp_elems_in_range_container)
                global_max_subset, global_max_iscore = keep_max_elements_with_range(local_sample, iscore, error_range)  # This method has static variables


                # if abs(iscore - global_max_iscore) >= error_range:
                #     step = len(sample_star[1]) / 2
                #     if count_child < 2 or count_child % step == 0:
                #         last_candidates.append((iscore, local_sample))
                #     count_child += 1
                #     print 'last candidate len: ', len(last_candidates)
                # if abs(iscore - local_max_iscore) <= error_range or local_max_iscore == 0:
                if iscore >= local_max_iscore:
                    key = len(local_sample)
                    local_max_iscore = iscore
                    if not is_exist(global_dict, local_sample):
                        if key in global_dict:
                            global_dict[key].append(local_sample)
                        else:
                            global_dict[key] = [local_sample]

                        # last_candidates = [(iscore, local_sample)]
                    # if local_max_iscore == iscore:
                        last_candidates.append((iscore, local_sample))
                        print 'Last candidate len: ', len(last_candidates)
                        if 'Binding_percentage_Hfold_interacting' in local_sample or 'Delta_Delta_G' in local_sample:
                            print "iscore and local sample", iscore, 'len:', len(local_sample), local_sample





                        # #            print '\n', iscore, 'local_sample: len(',len(local_sample), ')', local_sample
    #             if abs(iscore - local_max_iscore) <= error_range:
    #                 local_max_subset.append((iscore, local_sample))
    #             elif iscore - local_max_iscore > error_range:
    #             # else:
    #                 local_max_iscore = iscore
    #                 local_max_subset = [(iscore, local_sample)]
    # #                print '\n------------------------------------\n', local_max_iscore, 'local_max:', local_max_subset, '\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'

        #Drop a variable (i.e., each sample has length of one less than the original)
        # sample_star = local_max_subset[-1][1]
        # sample_star = local_max_subset
        # all_sample_stars = sample_uniformly(last_candidates, 10)
        all_sample_stars = last_candidates

       
#         #Keep the best I-Score
#         print "local_max_iscore: ", local_max_iscore, "global_max_iscore: ", global_max_iscore, "difference: ", abs(local_max_iscore - global_max_iscore)
#         if abs(local_max_iscore - global_max_iscore) <= error_range:
#             global_max_subset += local_max_subset
#         elif local_max_iscore - global_max_iscore > error_range:
#             global_max_iscore = local_max_iscore
#             global_max_subset = local_max_subset
# #            print "\n\n>>>>>>>>>>>>>>>>>>>> Global max updated <<<<<<<<<<<<<<<<<<<<"
# #            print '\n', global_max_iscore, 'local_sample: len(', len(global_max_subset), ')', global_max_subset
# ##        print 'global_max_subset', global_max_subset
# #     print 'global_max_iscore: ', global_max_iscore, ' global_max_subset: ', global_max_subset
    return global_max_iscore, global_max_subset


def feature_selection(data_frame, target_feature_name, initial_subset_len, bins_num, error_range, debug=False):
    max_iscore = -float('Inf')
    max_subsets = []
    # We copy the data_frame. Since the train data combination is different every
    # round, the discritization will be different and the changes remain in the
    # data frame (that's why we copy).
    df = data_frame.copy()

    df = convert_nominal_to_int(df)
    df = convert_normalized_to_discrete_equal_bin(df, bins_num)

    #Standard columns' name: starts with no number of special characters
    temp = {}
    for name in df.columns:
        temp[name] = correct_name(name)
    df = df.rename(columns=temp)

    # #Remove target column for creating the feature sets
    # df2 = df.copy(deep=True)
    # ######Check is the drop function creats a copy of df2 or use aliasing???
    # df2 = df2.drop(target_feature_name, 1)  # where 1 is the axis number (0 for rows and 1 for columns)
    column_lable_list = df.columns.values.tolist()
    #column_lable_list = [str(i) for i in df.columns.values.tolist()]
    if debug:
        print "target_feature_name: ", target_feature_name
        print column_lable_list.remove(target_feature_name)
        print "column_lable_list: ", column_lable_list

    column_lable_list.remove(target_feature_name)
    all_subsets = get_all_initial_subsets(column_lable_list, initial_subset_len)

    # all_subsets = get_all_initial_subsets(df2.columns, initial_subset_len)
#    all_subsets = [df2.columns]

    count = 0
    total_num = len(all_subsets)
    while len(all_subsets) > 0:
        s = all_subsets.pop()
    #for s in all_subsets:

        #Get the feature set with the highest I-Score according to the initial set 
        iscore, selected_sets = BDA(df, s, bins_num, target_feature_name, error_range)
        count += 1
        if abs(iscore - max_iscore) <= error_range:
            max_subsets += selected_sets
            if debug:
                print  '(', str(count), ' out of ', total_num, ') I-Score: ', iscore
                print 'max_subsets: ', max_subsets
        elif iscore - max_iscore > error_range:
            if debug:
                print '{0:.32f}'.format(iscore)
                print '{0:.32f}'.format(max_iscore)
                print '\n', 'iscore: ', str(iscore), '> max_iscore: ', str(max_iscore), iscore > max_iscore
            max_iscore = iscore
            max_subsets = selected_sets
            if debug:
                print  '(', str(count), ' out of ', total_num, ') I-Score: ', iscore
                print 'max_subsets: ', max_subsets
    if debug:
        print '\nInitial subset length: ', initial_subset_len
        print 'Best I-Score: ', max_iscore
    return max_subsets
    


if __name__ == '__main__':
    f_addr = '/home/seyedmah/Desktop/Machine learning in Muscular dystrophy/normalized_data_Jan10(Exon_Malueka_Category_C-0_A-1)_0.xls'
    target_feature_name = 'skip_percentage'
    initial_subset_len = 19
    bins_num = 6#It is fixed according to convert_normalized_to_discrete function
    error_range = 0.000000000000001

    data_frame = read_file(f_addr)
    max_subsets = feature_selection(data_frame, target_feature_name, initial_subset_len, bins_num, error_range)

    for i in range(len(max_subsets)):
        print '\n', i + 1, ' Best feature set len: ', len(max_subsets[i][1])
        print 'Best feature set: ', max_subsets[i]

    # error_range = 0.5
    # print keep_max_elements_with_range([20], 20, error_range)
    # print keep_max_elements_with_range([19.6], 19.6, error_range)
    # print keep_max_elements_with_range([19.7], 19.7, error_range)
    # print keep_max_elements_with_range([19.9], 19.9, error_range)
    # print keep_max_elements_with_range([20.4], 20.4, error_range)
    # print keep_max_elements_with_range([25], 25, error_range)
    # print keep_max_elements_with_range([24.5], 24.5, error_range)
    # print keep_max_elements_with_range([25.5], 25.5, error_range)
    # print keep_max_elements_with_range([25.3], 25.3, error_range)

