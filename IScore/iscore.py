import math
import sys
import traceback

#Y is a list of the target values
#y_avg is the global average of target values
#total_num is the total number of input records (X, y) where X is vector of explanatory variable and y is the# dependent/taregt variable

def compute_iscore(Y, Y_cells_avg):
    total_num = len(Y)
    assert total_num != 0

    y_sum = 0
    for y in Y:
        y_sum += y
    y_avg = float(y_sum)/len(Y)
        

    num = 0
    for inx in Y_cells_avg:
        num += pow(Y_cells_avg[inx], 2) * pow((Y_cells_avg[inx] - y_avg), 2)
    
    denom = 0
    for i in range(total_num):
        denom += pow((Y[i] - y_avg), 2)

    iscore = float(num)/denom
    if math.isnan(iscore):
        print "Numerator: ", num, "Denominator: ", denom
        print "Y: ", Y
        print "Y_cells_avg: ", Y_cells_avg
        print("Error: I-Score is NaN!")
        traceback.print_stack()
        exit(1)

    return iscore


